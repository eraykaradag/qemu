/*
 * Execution-driven runahead prefetching for RISC-V / KVM postcopy migration.
 *
 * When the first page fault fires on the destination, this module launches
 * a shadow thread that walks the guest instruction stream forward, simulates
 * ALU operations to track register state, and pre-requests pages for upcoming
 * Load/Store instructions before the vCPU ever faults on them.
 *
 * Copyright (c) 2024 Ahmet Eray Karadag
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/thread.h"
#include "qemu/rcu.h"
#include "qemu/error-report.h"
#include "system/kvm.h"
#include "system/memory.h"
#include "system/address-spaces.h"
#include "system/ramblock.h"
#include "exec/cpu-common.h"
#include "exec/target_page.h"
#include "migration/postcopy-ram.h"
#include "migration/migration.h"
#include "hw/core/cpu.h"

#define RUNAHEAD_MAX_INSNS  2048

/*
 * RISC-V KVM one-register IDs (from linux-headers/asm-riscv/kvm.h):
 *   id = KVM_REG_RISCV | KVM_REG_SIZE_U64 | type | index
 *
 * kvm_riscv_core layout: [0]=PC, [1..31]=x1..x31
 * kvm_riscv_csr  layout: [0]=sstatus … [8]=satp
 */
#define RA_KVM_RISCV    0x8000000000000000ULL
#define RA_KVM_SIZE_U64 0x0030000000000000ULL
#define RA_CORE_REG(i)  (RA_KVM_RISCV | RA_KVM_SIZE_U64 | (0x02ULL << 24) | (uint64_t)(i))
#define RA_CSR_REG(i)   (RA_KVM_RISCV | RA_KVM_SIZE_U64 | (0x03ULL << 24) | (uint64_t)(i))
#define RA_REG_PC       RA_CORE_REG(0)
#define RA_REG_SATP     RA_CSR_REG(8)   /* byte-offset 64/8 in kvm_riscv_csr */

typedef struct {
    QemuThread              thread;
    uint64_t                regs[32];   /* shadow GPR file, x0..x31         */
    uint64_t                reg_valid;  /* bitmask: bit i=1 → regs[i] valid */
    uint64_t                pc;
    uint64_t                satp;
    CPUState               *cs;
    MigrationIncomingState *mis;
} RunaheadState;

/* ---------- shadow register helpers -------------------------------- */

static inline bool ra_valid(RunaheadState *s, int r)
{
    return !!(s->reg_valid & (1ULL << r));
}

static inline void ra_write(RunaheadState *s, int rd, uint64_t v)
{
    if (rd == 0) return;
    s->regs[rd]   = v;
    s->reg_valid |= 1ULL << rd;
}

static inline void ra_inv(RunaheadState *s, int rd)
{
    if (rd == 0) return;
    s->reg_valid &= ~(1ULL << rd);
}

static inline int64_t ra_sext(uint64_t v, int bits)
{
    int sh = 64 - bits;
    return (int64_t)(v << sh) >> sh;
}

/* ---------- CPU state snapshot via KVM ----------------------------- */

/*
 * The faulted vCPU is blocked inside the kernel (userfaultfd path within
 * KVM_RUN). Its register state is still live in KVM, so we read it
 * directly via kvm_get_one_reg() without requiring a QEMU env sync.
 */
static bool runahead_snapshot_registers(RunaheadState *s)
{
    uint64_t reg;
    int i;

    if (kvm_get_one_reg(s->cs, RA_REG_PC, &reg))
        return false;
    s->pc = reg;

    s->regs[0]   = 0;
    s->reg_valid = ~0ULL;   /* all registers valid at snapshot time */

    for (i = 1; i < 32; i++) {
        if (kvm_get_one_reg(s->cs, RA_CORE_REG(i), &reg))
            return false;
        s->regs[i] = reg;
    }

    if (kvm_get_one_reg(s->cs, RA_REG_SATP, &reg))
        return false;
    s->satp = reg;

    return true;
}

/* ---------- Sv39 page table walk: GVA → GPA ----------------------- */

static uint64_t runahead_sv39_walk(uint64_t satp, uint64_t va)
{
    int mode = (satp >> 60) & 0xF;

    if (mode == 0)
        return va;      /* Bare – no translation */
    if (mode != 8)
        return -1ULL;   /* Only Sv39 for now; TODO: Sv48/57 */

    uint64_t vpn[3] = {
        (va >> 12) & 0x1FF,
        (va >> 21) & 0x1FF,
        (va >> 30) & 0x1FF,
    };
    uint64_t pt = (satp & ((1ULL << 44) - 1)) << 12;

    for (int lvl = 2; lvl >= 0; lvl--) {
        uint64_t pte = 0;
        cpu_physical_memory_read(pt + vpn[lvl] * 8, &pte, sizeof(pte));

        if (!(pte & 1))
            return -1ULL;   /* V bit clear */

        uint64_t ppn = (pte >> 10) & ((1ULL << 44) - 1);

        if (pte & 0xE) {    /* R|W|X set → leaf PTE */
            uint64_t pa = ppn << 12;
            for (int i = 0; i < lvl; i++)
                pa |= vpn[i] << (12 + 9 * i);  /* superpage lower bits */
            return pa | (va & 0xFFF);
        }

        pt = ppn << 12;
    }
    return -1ULL;
}

/* ---------- GVA → page request ------------------------------------- */

static void runahead_prefetch(RunaheadState *s, uint64_t gva)
{
    uint64_t gpa = runahead_sv39_walk(s->satp, gva);
    if (gpa == -1ULL)
        return;

    RCU_READ_LOCK_GUARD();
    hwaddr xlat, len = TARGET_PAGE_SIZE;
    MemoryRegion *mr = address_space_translate(&address_space_memory,
                                               gpa, &xlat, &len,
                                               false, MEMTXATTRS_UNSPECIFIED);
    if (!memory_region_is_ram(mr))
        return;

    void *hva = qemu_map_ram_ptr(mr->ram_block, xlat);

    ram_addr_t rb_offset;
    RAMBlock *rb = qemu_ram_block_from_host(hva, true, &rb_offset);
    if (!rb)
        return;

    postcopy_runahead_prefetch_page(s->mis, rb, rb_offset,
                                    (uint64_t)(uintptr_t)hva);
}

/* ---------- instruction simulation --------------------------------- */

/*
 * Simulate one 32-bit RISC-V instruction against the shadow state.
 * Returns next PC, or 0 to stop runahead.
 */
static uint64_t runahead_sim_insn32(RunaheadState *s, uint32_t insn)
{
    int op = insn & 0x7F;
    int rd  = (insn >> 7)  & 0x1F;
    int rs1 = (insn >> 15) & 0x1F;
    int rs2 = (insn >> 20) & 0x1F;
    int f3  = (insn >> 12) & 0x7;
    int f7  = (insn >> 25) & 0x7F;
    uint64_t next_pc = s->pc + 4;

    switch (op) {

    case 0x03: /* LOAD */
    case 0x07: /* LOAD-FP */ {
        int64_t imm = ra_sext(insn >> 20, 12);
        if (ra_valid(s, rs1))
            runahead_prefetch(s, s->regs[rs1] + imm);
        if (op == 0x03)
            ra_inv(s, rd);  /* value unknown until page arrives */
        break;
    }

    case 0x23: /* STORE */
    case 0x27: /* STORE-FP */ {
        int64_t imm = ra_sext(((insn >> 25) << 5) | ((insn >> 7) & 0x1F), 12);
        if (ra_valid(s, rs1))
            runahead_prefetch(s, s->regs[rs1] + imm);
        break;
    }

    case 0x2F: /* AMO – address is rs1 directly */
        if (ra_valid(s, rs1))
            runahead_prefetch(s, s->regs[rs1]);
        ra_inv(s, rd);
        break;

    case 0x37: /* LUI */
        ra_write(s, rd, (int64_t)(int32_t)(insn & 0xFFFFF000));
        break;

    case 0x17: /* AUIPC */
        ra_write(s, rd, s->pc + (int64_t)(int32_t)(insn & 0xFFFFF000));
        break;

    case 0x13: /* OP-IMM */ {
        if (!ra_valid(s, rs1)) { ra_inv(s, rd); break; }
        uint64_t a = s->regs[rs1];
        int64_t  imm   = ra_sext(insn >> 20, 12);
        uint64_t shamt = (insn >> 20) & 0x3F;
        uint64_t res;
        switch (f3) {
        case 0: res = a + imm;                                        break;
        case 1: res = a << shamt;                                     break;
        case 2: res = (int64_t)a < imm ? 1 : 0;                      break;
        case 3: res = a < (uint64_t)(int64_t)imm ? 1 : 0;            break;
        case 4: res = a ^ (uint64_t)imm;                              break;
        case 5: res = (f7 & 0x20) ? (uint64_t)((int64_t)a >> shamt)
                                  : a >> shamt;                       break;
        case 6: res = a | (uint64_t)imm;                              break;
        case 7: res = a & (uint64_t)imm;                              break;
        default: ra_inv(s, rd); goto done;
        }
        ra_write(s, rd, res);
        break;
    }

    case 0x1B: /* OP-IMM-32 */ {
        if (!ra_valid(s, rs1)) { ra_inv(s, rd); break; }
        uint64_t a = s->regs[rs1];
        int64_t  imm   = ra_sext(insn >> 20, 12);
        uint64_t shamt = (insn >> 20) & 0x1F;
        uint64_t res;
        switch (f3) {
        case 0: res = (int64_t)(int32_t)((uint32_t)a + (int32_t)imm); break;
        case 1: res = (int64_t)(int32_t)((uint32_t)a << shamt);        break;
        case 5: res = (f7 & 0x20)
                      ? (int64_t)((int32_t)a >> shamt)
                      : (int64_t)(int32_t)((uint32_t)a >> shamt);      break;
        default: ra_inv(s, rd); goto done;
        }
        ra_write(s, rd, res);
        break;
    }

    case 0x33: /* OP */ {
        if (!ra_valid(s, rs1) || !ra_valid(s, rs2)) { ra_inv(s, rd); break; }
        if (f7 == 0x01) { ra_inv(s, rd); break; }  /* M ext: MUL/DIV/REM */
        uint64_t a = s->regs[rs1], b = s->regs[rs2];
        uint64_t res;
        switch (f3) {
        case 0: res = (f7 & 0x20) ? a - b : a + b;                   break;
        case 1: res = a << (b & 0x3F);                                break;
        case 2: res = (int64_t)a < (int64_t)b ? 1 : 0;               break;
        case 3: res = a < b ? 1 : 0;                                  break;
        case 4: res = a ^ b;                                          break;
        case 5: res = (f7 & 0x20) ? (uint64_t)((int64_t)a >> (b & 0x3F))
                                  : a >> (b & 0x3F);                  break;
        case 6: res = a | b;                                          break;
        case 7: res = a & b;                                          break;
        default: ra_inv(s, rd); goto done;
        }
        ra_write(s, rd, res);
        break;
    }

    case 0x3B: /* OP-32 */ {
        if (!ra_valid(s, rs1) || !ra_valid(s, rs2)) { ra_inv(s, rd); break; }
        if (f7 == 0x01) { ra_inv(s, rd); break; }  /* M ext: MULW/DIVW/REMW */
        uint64_t a = s->regs[rs1], b = s->regs[rs2];
        uint64_t res;
        switch (f3) {
        case 0: res = (int64_t)(int32_t)((f7 & 0x20) ? a - b : a + b);    break;
        case 1: res = (int64_t)(int32_t)((uint32_t)a << (b & 0x1F));       break;
        case 5: res = (f7 & 0x20)
                      ? (int64_t)((int32_t)a >> (b & 0x1F))
                      : (int64_t)(int32_t)((uint32_t)a >> (b & 0x1F));     break;
        default: ra_inv(s, rd); goto done;
        }
        ra_write(s, rd, res);
        break;
    }

    case 0x6F: /* JAL */ {
        int64_t imm = ra_sext(
            (((insn >> 31) & 1) << 20) | (((insn >> 12) & 0xFF) << 12) |
            (((insn >> 20) & 1) << 11) | (((insn >> 21) & 0x3FF) << 1), 21);
        ra_write(s, rd, s->pc + 4);
        next_pc = s->pc + imm;
        break;
    }

    case 0x67: /* JALR */ {
        int64_t imm = ra_sext(insn >> 20, 12);
        ra_write(s, rd, s->pc + 4);
        if (!ra_valid(s, rs1))
            return 0;   /* unknown target – stop runahead */
        next_pc = (s->regs[rs1] + imm) & ~1ULL;
        break;
    }

    case 0x63: /* BRANCH – backward-taken heuristic */ {
        int64_t imm = ra_sext(
            (((insn >> 31) & 1) << 12) | (((insn >> 7)  & 1) << 11) |
            (((insn >> 25) & 0x3F) << 5) | (((insn >> 8) & 0xF) << 1), 13);
        if (imm < 0)
            next_pc = s->pc + imm;  /* backward → likely loop back-edge */
        break;
    }

    case 0x0F: /* FENCE */
    case 0x73: /* SYSTEM / CSR */
        break;

    default:
        return 0;
    }

done:
    return next_pc;
}

/* ---------- thread entry point ------------------------------------- */

static void *runahead_thread_routine(void *opaque)
{
    RunaheadState *s = (RunaheadState *)opaque;
    int i;

    rcu_register_thread();
    fprintf(stderr, "[RUNAHEAD] Thread started\n");

    if (!runahead_snapshot_registers(s)) {
        fprintf(stderr, "[RUNAHEAD] Failed to read KVM registers\n");
        goto out;
    }
    fprintf(stderr, "[RUNAHEAD] PC=0x%" PRIx64 " satp=0x%" PRIx64 "\n",
            s->pc, s->satp);

    for (i = 0; i < RUNAHEAD_MAX_INSNS; i++) {
        uint64_t insn_gpa = runahead_sv39_walk(s->satp, s->pc);
        if (insn_gpa == -1ULL) {
            fprintf(stderr, "[RUNAHEAD] Unmapped PC 0x%" PRIx64 "\n", s->pc);
            break;
        }

        uint32_t insn = 0;
        cpu_physical_memory_read(insn_gpa, &insn, sizeof(insn));

        /* 16-bit compressed instruction – advance PC without simulating */
        if ((insn & 0x3) != 0x3) {
            s->pc += 2;
            continue;
        }

        uint64_t next_pc = runahead_sim_insn32(s, insn);
        if (!next_pc)
            break;
        s->pc = next_pc;
    }

    fprintf(stderr, "[RUNAHEAD] Thread done after %d insns\n", i);
out:
    rcu_unregister_thread();
    g_free(s);
    return NULL;
}

/* ---------- arch hook – called from postcopy-ram.c ----------------- */

void postcopy_runahead_arch_start(CPUState *cs, MigrationIncomingState *mis)
{
    RunaheadState *s = g_new0(RunaheadState, 1);
    s->cs  = cs;
    s->mis = mis;
    qemu_thread_create(&s->thread, "runahead_thread",
                       runahead_thread_routine, s,
                       QEMU_THREAD_DETACHED);
}
