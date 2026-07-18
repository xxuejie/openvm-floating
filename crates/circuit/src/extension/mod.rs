use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, VmCircuitExtension, VmExecutionExtension,
        VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::AnyEnum;
use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_instructions::{riscv::RV32_REGISTER_AS, LocalOpcode};
use openvm_rv32f_transpiler::FpOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine, StarkProtocolConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    adapters::*, fadd::*, fclass::*, fcmp::*, fcvt_f2i::*, fcvt_i2f::*, fdiv::*, fma::*,
    fminmax::*, fmul::*, fsgnj::*, fsqrt::*, fsub::*,
};

/// RISC-V F (single-precision floating-point) extension config.
///
/// Supports both F extension and Zfinx mode via `fp_register_as`:
/// - `1` (default): Zfinx — floats in integer registers (AS 1)
/// - `5`: F extension — separate FP register file (AS 5)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Rv32F {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
    #[serde(default = "default_fp_register_as")]
    pub fp_register_as: u32,
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 8 * (1 << 8)]
}

fn default_fp_register_as() -> u32 {
    1 // Zfinx: integer register file
}

impl Default for Rv32F {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
            fp_register_as: default_fp_register_as(),
        }
    }
}

/// F-extension executor enum.
#[derive(
    Clone,
    From,
    AnyEnum,
    openvm_circuit_derive::Executor,
    openvm_circuit_derive::MeteredExecutor,
    openvm_circuit_derive::PreflightExecutor,
)]
pub enum Rv32FExecutor {
    Fmul(Rv32FmulExecutor),
    Fadd(Rv32FaddExecutor),
    Fsub(Rv32FsubExecutor),
    Fsgnj(Rv32FsgnjExecutor),
    Fminmax(Rv32FminmaxExecutor),
    Fcmp(Rv32FcmpExecutor),
    Fclass(Rv32FclassExecutor),
    FcvtF2I(Rv32FcvtF2IExecutor),
    FcvtI2F(Rv32FcvtI2FExecutor),
    Fma(Rv32FmaExecutor),
    Fdiv(Rv32FdivExecutor),
    Fsqrt(Rv32FsqrtExecutor),
}

impl<F: PrimeField32> VmExecutionExtension<F> for Rv32F {
    type Executor = Rv32FExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv32FExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let adapter_exec = Rv32FpAdapterExecutor::new(self.fp_register_as);
        let fmul = Rv32FmulExecutor::new(adapter_exec, FpOpcode::CLASS_OFFSET, self.fp_register_as);
        inventory.add_executor(fmul, [FpOpcode::FMUL_S.global_opcode()])?;

        let adapter_exec = Rv32FpAdapterExecutor::new(self.fp_register_as);
        let fadd = Rv32FaddExecutor::new(adapter_exec, FpOpcode::CLASS_OFFSET, self.fp_register_as);
        inventory.add_executor(fadd, [FpOpcode::FADD_S.global_opcode()])?;

        let adapter_exec = Rv32FpAdapterExecutor::new(self.fp_register_as);
        let fdiv = Rv32FdivExecutor::new(adapter_exec, FpOpcode::CLASS_OFFSET, self.fp_register_as);
        inventory.add_executor(fdiv, [FpOpcode::FDIV_S.global_opcode()])?;

        let adapter_exec =
            Rv32FpUnaryAdapterExecutor::new(self.fp_register_as, self.fp_register_as);
        let fsqrt = Rv32FsqrtExecutor::new(
            adapter_exec,
            FpOpcode::CLASS_OFFSET,
            self.fp_register_as,
            self.fp_register_as,
        );
        inventory.add_executor(fsqrt, [FpOpcode::FSQRT_S.global_opcode()])?;

        let adapter_exec = Rv32FpAdapterExecutor::new(self.fp_register_as);
        let fsub = Rv32FsubExecutor::new(adapter_exec, FpOpcode::CLASS_OFFSET, self.fp_register_as);
        inventory.add_executor(fsub, [FpOpcode::FSUB_S.global_opcode()])?;

        let adapter_exec = Rv32FpAdapterExecutor::new(self.fp_register_as);
        let fsgnj =
            Rv32FsgnjExecutor::new(adapter_exec, FpOpcode::CLASS_OFFSET, self.fp_register_as);
        inventory.add_executor(
            fsgnj,
            [
                FpOpcode::FSGNJ_S.global_opcode(),
                FpOpcode::FSGNJN_S.global_opcode(),
                FpOpcode::FSGNJX_S.global_opcode(),
            ],
        )?;

        let adapter_exec = Rv32FpAdapterExecutor::new(self.fp_register_as);
        let fminmax =
            Rv32FminmaxExecutor::new(adapter_exec, FpOpcode::CLASS_OFFSET, self.fp_register_as);
        inventory.add_executor(
            fminmax,
            [
                FpOpcode::FMIN_S.global_opcode(),
                FpOpcode::FMAX_S.global_opcode(),
            ],
        )?;

        let adapter_exec = Rv32FcmpAdapterExecutor::new(self.fp_register_as);
        let fcmp = Rv32FcmpExecutor::new(adapter_exec, FpOpcode::CLASS_OFFSET, self.fp_register_as);
        inventory.add_executor(
            fcmp,
            [
                FpOpcode::FLE_S.global_opcode(),
                FpOpcode::FLT_S.global_opcode(),
                FpOpcode::FEQ_S.global_opcode(),
            ],
        )?;

        let adapter_exec = Rv32FpUnaryAdapterExecutor::new(self.fp_register_as, RV32_REGISTER_AS);
        let fclass = Rv32FclassExecutor::new(
            adapter_exec,
            FpOpcode::CLASS_OFFSET,
            self.fp_register_as,
            RV32_REGISTER_AS,
        );
        inventory.add_executor(fclass, [FpOpcode::FCLASS_S.global_opcode()])?;

        let adapter_exec = Rv32FpUnaryAdapterExecutor::new(self.fp_register_as, RV32_REGISTER_AS);
        let fcvt_f2i = Rv32FcvtF2IExecutor::new(
            adapter_exec,
            FpOpcode::CLASS_OFFSET,
            self.fp_register_as,
            RV32_REGISTER_AS,
        );
        inventory.add_executor(
            fcvt_f2i,
            [
                FpOpcode::FCVT_W_S.global_opcode(),
                FpOpcode::FCVT_WU_S.global_opcode(),
            ],
        )?;

        let adapter_exec = Rv32FpUnaryAdapterExecutor::new(RV32_REGISTER_AS, self.fp_register_as);
        let fcvt_i2f = Rv32FcvtI2FExecutor::new(
            adapter_exec,
            FpOpcode::CLASS_OFFSET,
            RV32_REGISTER_AS,
            self.fp_register_as,
        );
        inventory.add_executor(
            fcvt_i2f,
            [
                FpOpcode::FCVT_S_W.global_opcode(),
                FpOpcode::FCVT_S_WU.global_opcode(),
            ],
        )?;

        let adapter_exec = Rv32Fp3AdapterExecutor::new(self.fp_register_as);
        let fma = Rv32FmaExecutor::new(adapter_exec, FpOpcode::CLASS_OFFSET, self.fp_register_as);
        inventory.add_executor(
            fma,
            [
                FpOpcode::FMADD_S.global_opcode(),
                FpOpcode::FMSUB_S.global_opcode(),
                FpOpcode::FNMSUB_S.global_opcode(),
                FpOpcode::FNMADD_S.global_opcode(),
            ],
        )?;
        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Rv32F {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();
        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker_bus = inventory.range_checker().bus;

        let range_tuple_checker = {
            let existing_air = inventory.find_air::<RangeTupleCheckerAir<2>>().find(|c| {
                c.bus.sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus.sizes[1] >= self.range_tuple_checker_sizes[1]
            });
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = RangeTupleCheckerBus::new(
                    inventory.new_bus_idx(),
                    self.range_tuple_checker_sizes,
                );
                let air = RangeTupleCheckerAir { bus };
                inventory.add_air(air);
                air.bus
            }
        };

        let fmul = Rv32FmulAir::new(
            Rv32FpAdapterAir::new(exec_bridge, memory_bridge, self.fp_register_as),
            FmulCoreAir::new(range_tuple_checker, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fmul);

        let fadd = Rv32FaddAir::new(
            Rv32FpAdapterAir::new(exec_bridge, memory_bridge, self.fp_register_as),
            FaddCoreAir::new(range_tuple_checker, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fadd);

        let fdiv = Rv32FdivAir::new(
            Rv32FpAdapterAir::new(exec_bridge, memory_bridge, self.fp_register_as),
            FdivCoreAir::new(range_checker_bus, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fdiv);

        let fsqrt = Rv32FsqrtAir::new(
            Rv32FpUnaryAdapterAir::new(
                exec_bridge,
                memory_bridge,
                self.fp_register_as,
                self.fp_register_as,
            ),
            FsqrtCoreAir::new(range_checker_bus, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fsqrt);

        let fsub = Rv32FsubAir::new(
            Rv32FpAdapterAir::new(exec_bridge, memory_bridge, self.fp_register_as),
            FsubCoreAir::new(range_tuple_checker, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fsub);

        let fsgnj = Rv32FsgnjAir::new(
            Rv32FpAdapterAir::new(exec_bridge, memory_bridge, self.fp_register_as),
            FsgnjCoreAir::new(FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fsgnj);

        let fminmax = Rv32FminmaxAir::new(
            Rv32FpAdapterAir::new(exec_bridge, memory_bridge, self.fp_register_as),
            FminmaxCoreAir::new(range_checker_bus, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fminmax);

        let fcmp = Rv32FcmpAir::new(
            Rv32FcmpAdapterAir::new(exec_bridge, memory_bridge, self.fp_register_as),
            FcmpCoreAir::new(range_checker_bus, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fcmp);

        let fclass = Rv32FclassAir::new(
            Rv32FpUnaryAdapterAir::new(
                exec_bridge,
                memory_bridge,
                self.fp_register_as,
                RV32_REGISTER_AS,
            ),
            FclassCoreAir::new(range_checker_bus, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fclass);

        let fcvt_f2i = Rv32FcvtF2IAir::new(
            Rv32FpUnaryAdapterAir::new(
                exec_bridge,
                memory_bridge,
                self.fp_register_as,
                RV32_REGISTER_AS,
            ),
            FcvtF2ICoreAir::new(range_checker_bus, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fcvt_f2i);

        let fcvt_i2f = Rv32FcvtI2FAir::new(
            Rv32FpUnaryAdapterAir::new(
                exec_bridge,
                memory_bridge,
                RV32_REGISTER_AS,
                self.fp_register_as,
            ),
            FcvtI2FCoreAir::new(range_checker_bus, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fcvt_i2f);

        let fma = Rv32FmaAir::new(
            Rv32Fp3AdapterAir::new(exec_bridge, memory_bridge, self.fp_register_as),
            FmaCoreAir::new(range_checker_bus, FpOpcode::CLASS_OFFSET),
        );
        inventory.add_air(fma);

        Ok(())
    }
}

/// CPU prover extension for F extension.
pub struct Rv32FCpuProverExt;

impl<SC, E, RA> VmProverExtension<E, RA, Rv32F> for Rv32FCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: openvm_circuit::arch::RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &Rv32F,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<SharedRangeTupleCheckerChip<2>>()
                .find(|c| {
                    c.bus().sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.bus().sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &RangeTupleCheckerAir<2> = inventory.next_air()?;
                let chip = SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Rv32FmulAir>()?;
        let fmul = Rv32FmulChip::new(
            FmulFiller::new(
                Rv32FpAdapterFiller,
                range_tuple_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fmul);

        inventory.next_air::<Rv32FaddAir>()?;
        let fadd = Rv32FaddChip::new(
            FaddFiller::new(
                Rv32FpAdapterFiller,
                range_tuple_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fadd);

        inventory.next_air::<Rv32FdivAir>()?;
        let fdiv = Rv32FdivChip::new(
            FdivFiller::new(
                Rv32FpAdapterFiller,
                range_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fdiv);

        inventory.next_air::<Rv32FsqrtAir>()?;
        let fsqrt = Rv32FsqrtChip::new(
            FsqrtFiller::new(
                Rv32FpUnaryAdapterFiller,
                range_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fsqrt);

        inventory.next_air::<Rv32FsubAir>()?;
        let fsub = Rv32FsubChip::new(
            FsubFiller::new(
                Rv32FpAdapterFiller,
                range_tuple_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fsub);

        inventory.next_air::<Rv32FsgnjAir>()?;
        let fsgnj = Rv32FsgnjChip::new(
            FsgnjFiller::new(Rv32FpAdapterFiller, FpOpcode::CLASS_OFFSET),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fsgnj);

        inventory.next_air::<Rv32FminmaxAir>()?;
        let fminmax = Rv32FminmaxChip::new(
            FminmaxFiller::new(
                Rv32FpAdapterFiller,
                range_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fminmax);

        inventory.next_air::<Rv32FcmpAir>()?;
        let fcmp = Rv32FcmpChip::new(
            FcmpFiller::new(
                Rv32FcmpAdapterFiller,
                range_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fcmp);

        inventory.next_air::<Rv32FclassAir>()?;
        let fclass = Rv32FclassChip::new(
            FclassFiller::new(
                Rv32FpUnaryAdapterFiller,
                range_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fclass);

        inventory.next_air::<Rv32FcvtF2IAir>()?;
        let fcvt_f2i = Rv32FcvtF2IChip::new(
            FcvtF2IFiller::new(
                Rv32FpUnaryAdapterFiller,
                range_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fcvt_f2i);

        inventory.next_air::<Rv32FcvtI2FAir>()?;
        let fcvt_i2f = Rv32FcvtI2FChip::new(
            FcvtI2FFiller::new(
                Rv32FpUnaryAdapterFiller,
                range_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fcvt_i2f);

        inventory.next_air::<Rv32FmaAir>()?;
        let fma = Rv32FmaChip::new(
            FmaFiller::new(
                Rv32Fp3AdapterFiller,
                range_checker.clone(),
                FpOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(fma);

        Ok(())
    }
}
