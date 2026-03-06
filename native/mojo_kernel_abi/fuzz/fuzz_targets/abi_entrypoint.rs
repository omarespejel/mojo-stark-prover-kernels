#![no_main]

use libfuzzer_sys::fuzz_target;
use mojo_kernel_abi::fuzz_abi_entrypoint;

fuzz_target!(|data: &[u8]| {
    let _ = fuzz_abi_entrypoint(data);
});
