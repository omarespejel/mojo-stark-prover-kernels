use std::ffi::c_char;
use std::mem;
use std::slice;

use rayon::prelude::*;

const DOMAIN_TAG: &[u8] = b"B2S_LAYER_V1";
const PERSONALIZATION: &[u8] = b"MSPKv1";
const HASH_SIZE: usize = 32;
const MAX_LOG_SIZE: u32 = 20;
const MAX_COLUMNS: u32 = 4096;
const MAX_TOTAL_CELLS: u64 = 200_000;
const MAX_DEBUG_BUFFER_LEN: u32 = 1_000_000;
const M31_PRIME: u32 = 2_147_483_647;
const MAX_M31_VECTOR_LEN: u32 = 1_000_000;
const MOJO_KERNEL_ABI_VERSION: u32 = 1;

const RC_OK: i32 = 0;
const RC_NULL_POINTER: i32 = 1;
const RC_INVALID_ARGUMENT: i32 = 2;
const RC_OVERFLOW: i32 = 3;
const RC_PANIC: i32 = 99;

const FUZZ_MAX_BUFFER_BYTES: u32 = 512 * 1024;
const FUZZ_MAX_CELLS: usize = 65_536;
const PARALLEL_MIN_ROWS: u32 = 512;
const PARALLEL_MIN_CELLS: u64 = 32_768;
const PARALLEL_MIN_CHUNK_ROWS: usize = 32;
const PARALLEL_MIN_M31_LEN: usize = 16_384;

const FLAG_NULL_COLUMNS: u32 = 1 << 0;
const FLAG_NULL_OUT: u32 = 1 << 1;
const FLAG_NULL_PREV: u32 = 1 << 2;
const FLAG_NULL_DEBUG: u32 = 1 << 3;
const FLAG_ALIAS_OUT_COLUMNS: u32 = 1 << 4;
const FLAG_ALIAS_OUT_PREV: u32 = 1 << 5;
const FLAG_ALIAS_OUT_DEBUG: u32 = 1 << 6;
const FLAG_MISALIGN_COLUMNS: u32 = 1 << 7;
const FLAG_OVERSIZE_DEBUG_LEN: u32 = 1 << 8;
const FLAG_FORCE_VALID_SHAPE: u32 = 1 << 9;

/// Returns the ABI version expected by host-side loaders.
#[no_mangle]
pub extern "C" fn mojo_kernel_abi_version() -> u32 {
    MOJO_KERNEL_ABI_VERSION
}

/// Computes per-row BLAKE2s commitments for one layer through a C ABI entrypoint.
///
/// # Safety
/// Caller must pass valid pointers and lengths for all non-null buffers according to the
/// ABI contract. In particular, `columns_flat` and `out_hashes` must point to writable/readable
/// memory regions of the requested lengths, and buffers must not alias in forbidden ways.
#[no_mangle]
pub unsafe extern "C" fn mojo_blake2s_commit_layer(
    log_size: u32,
    prev_layer_bytes: *const u8,
    prev_layer_len: u32,
    columns_flat: *const u32,
    n_columns: u32,
    n_rows: u32,
    out_hashes: *mut u8,
    out_hashes_len: u32,
    debug_level: u32,
    debug_buffer: *mut c_char,
    debug_buffer_len: u32,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        commit_impl(
            log_size,
            prev_layer_bytes,
            prev_layer_len,
            columns_flat,
            n_columns,
            n_rows,
            out_hashes,
            out_hashes_len,
            debug_level,
            debug_buffer,
            debug_buffer_len,
        )
    });
    match result {
        Ok(rc) => rc,
        Err(_) => {
            write_debug(debug_buffer, debug_buffer_len, "panic in kernel");
            RC_PANIC
        }
    }
}

/// Computes one M31 batch linear-combination through a C ABI entrypoint.
///
/// out[i] = (alpha * a[i] + beta * b[i] + c[i]) mod p, where p = 2^31 - 1.
///
/// # Safety
/// Caller must provide valid non-overlapping input/output buffers when non-null pointers are used.
#[no_mangle]
pub unsafe extern "C" fn mojo_m31_axpy(
    a: *const u32,
    b: *const u32,
    c: *const u32,
    len: u32,
    alpha: u32,
    beta: u32,
    out: *mut u32,
    out_len: u32,
    debug_level: u32,
    debug_buffer: *mut c_char,
    debug_buffer_len: u32,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        m31_axpy_impl(
            a,
            b,
            c,
            len,
            alpha,
            beta,
            out,
            out_len,
            debug_level,
            debug_buffer,
            debug_buffer_len,
        )
    });
    match result {
        Ok(rc) => rc,
        Err(_) => {
            write_debug(debug_buffer, debug_buffer_len, "panic in m31 kernel");
            RC_PANIC
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn m31_axpy_impl(
    a: *const u32,
    b: *const u32,
    c: *const u32,
    len: u32,
    alpha: u32,
    beta: u32,
    out: *mut u32,
    out_len: u32,
    debug_level: u32,
    debug_buffer: *mut c_char,
    debug_buffer_len: u32,
) -> i32 {
    if debug_buffer_len > MAX_DEBUG_BUFFER_LEN {
        return RC_INVALID_ARGUMENT;
    }

    if a.is_null() || b.is_null() || c.is_null() || out.is_null() {
        write_debug(debug_buffer, debug_buffer_len, "null m31 pointer");
        return RC_NULL_POINTER;
    }
    if len == 0 || out_len == 0 {
        write_debug(debug_buffer, debug_buffer_len, "m31 len must be positive");
        return RC_INVALID_ARGUMENT;
    }
    if len != out_len {
        write_debug(debug_buffer, debug_buffer_len, "m31 out_len mismatch");
        return RC_INVALID_ARGUMENT;
    }
    if len > MAX_M31_VECTOR_LEN {
        write_debug(debug_buffer, debug_buffer_len, "m31 len exceeds configured max");
        return RC_INVALID_ARGUMENT;
    }
    if alpha >= M31_PRIME || beta >= M31_PRIME {
        write_debug(debug_buffer, debug_buffer_len, "m31 scalar outside field");
        return RC_INVALID_ARGUMENT;
    }

    if !(a as usize).is_multiple_of(mem::align_of::<u32>())
        || !(b as usize).is_multiple_of(mem::align_of::<u32>())
        || !(c as usize).is_multiple_of(mem::align_of::<u32>())
        || !(out as usize).is_multiple_of(mem::align_of::<u32>())
    {
        write_debug(debug_buffer, debug_buffer_len, "m31 pointer is not u32-aligned");
        return RC_INVALID_ARGUMENT;
    }

    let len_usize = len as usize;
    if len_usize > isize::MAX as usize / mem::size_of::<u32>() {
        write_debug(debug_buffer, debug_buffer_len, "m31 len exceeds host limits");
        return RC_OVERFLOW;
    }

    let bytes_len = len_usize.checked_mul(mem::size_of::<u32>());
    let Some(bytes_len) = bytes_len else {
        write_debug(debug_buffer, debug_buffer_len, "m31 byte-size overflow");
        return RC_OVERFLOW;
    };

    let out_range = pointer_range(out as *const u8, bytes_len);
    let Some(out_range) = out_range else {
        write_debug(debug_buffer, debug_buffer_len, "m31 output pointer range overflow");
        return RC_OVERFLOW;
    };

    let a_range = pointer_range(a as *const u8, bytes_len);
    let Some(a_range) = a_range else {
        write_debug(debug_buffer, debug_buffer_len, "m31 a pointer range overflow");
        return RC_OVERFLOW;
    };
    let b_range = pointer_range(b as *const u8, bytes_len);
    let Some(b_range) = b_range else {
        write_debug(debug_buffer, debug_buffer_len, "m31 b pointer range overflow");
        return RC_OVERFLOW;
    };
    let c_range = pointer_range(c as *const u8, bytes_len);
    let Some(c_range) = c_range else {
        write_debug(debug_buffer, debug_buffer_len, "m31 c pointer range overflow");
        return RC_OVERFLOW;
    };

    if ranges_overlap(out_range, a_range)
        || ranges_overlap(out_range, b_range)
        || ranges_overlap(out_range, c_range)
    {
        write_debug(debug_buffer, debug_buffer_len, "output overlaps input");
        return RC_INVALID_ARGUMENT;
    }

    if !debug_buffer.is_null() && debug_buffer_len > 0 {
        let debug_range = pointer_range(debug_buffer as *const u8, debug_buffer_len as usize);
        let Some(debug_range) = debug_range else {
            return RC_OVERFLOW;
        };
        if ranges_overlap(out_range, debug_range) {
            write_debug(
                debug_buffer,
                debug_buffer_len,
                "output overlaps debug buffer",
            );
            return RC_INVALID_ARGUMENT;
        }
    }

    let a = slice::from_raw_parts(a, len_usize);
    let b = slice::from_raw_parts(b, len_usize);
    let c = slice::from_raw_parts(c, len_usize);
    let out = slice::from_raw_parts_mut(out, len_usize);

    for ((av, bv), cv) in a.iter().zip(b.iter()).zip(c.iter()) {
        if *av >= M31_PRIME || *bv >= M31_PRIME || *cv >= M31_PRIME {
            write_debug(debug_buffer, debug_buffer_len, "input value outside m31 field");
            return RC_INVALID_ARGUMENT;
        }
    }

    let use_parallel = should_use_parallel_m31(len_usize);
    if use_parallel {
        out.par_iter_mut()
            .with_min_len(PARALLEL_MIN_CHUNK_ROWS)
            .enumerate()
            .for_each(|(i, out_cell)| {
                *out_cell = m31_axpy_scalar(alpha, beta, a[i], b[i], c[i]);
            });
    } else {
        for i in 0..len_usize {
            out[i] = m31_axpy_scalar(alpha, beta, a[i], b[i], c[i]);
        }
    }

    if debug_level > 0 {
        let mode = if use_parallel { "parallel" } else { "serial" };
        let msg = format!("ok len={} mode={}", len, mode);
        write_debug(debug_buffer, debug_buffer_len, &msg);
    } else {
        write_debug(debug_buffer, debug_buffer_len, "");
    }
    RC_OK
}

#[allow(clippy::too_many_arguments)]
unsafe fn commit_impl(
    log_size: u32,
    prev_layer_bytes: *const u8,
    prev_layer_len: u32,
    columns_flat: *const u32,
    n_columns: u32,
    n_rows: u32,
    out_hashes: *mut u8,
    out_hashes_len: u32,
    debug_level: u32,
    debug_buffer: *mut c_char,
    debug_buffer_len: u32,
) -> i32 {
    if debug_buffer_len > MAX_DEBUG_BUFFER_LEN {
        return RC_INVALID_ARGUMENT;
    }

    if columns_flat.is_null() || out_hashes.is_null() {
        write_debug(debug_buffer, debug_buffer_len, "null columns/out pointer");
        return RC_NULL_POINTER;
    }

    if n_columns == 0 || n_rows == 0 {
        write_debug(debug_buffer, debug_buffer_len, "n_columns and n_rows must be positive");
        return RC_INVALID_ARGUMENT;
    }

    if log_size > MAX_LOG_SIZE {
        write_debug(debug_buffer, debug_buffer_len, "log_size exceeds configured max");
        return RC_INVALID_ARGUMENT;
    }

    if n_columns > MAX_COLUMNS {
        write_debug(debug_buffer, debug_buffer_len, "n_columns exceeds configured max");
        return RC_INVALID_ARGUMENT;
    }

    if !(columns_flat as usize).is_multiple_of(mem::align_of::<u32>()) {
        write_debug(debug_buffer, debug_buffer_len, "columns pointer is not u32-aligned");
        return RC_INVALID_ARGUMENT;
    }

    let expected_rows = 1u64 << log_size;
    if expected_rows != n_rows as u64 {
        write_debug(
            debug_buffer,
            debug_buffer_len,
            "n_rows must equal 2^log_size",
        );
        return RC_INVALID_ARGUMENT;
    }

    let total_cells = (n_rows as u64).checked_mul(n_columns as u64);
    let Some(total_cells) = total_cells else {
        write_debug(debug_buffer, debug_buffer_len, "cells overflow");
        return RC_OVERFLOW;
    };
    if total_cells > MAX_TOTAL_CELLS {
        write_debug(debug_buffer, debug_buffer_len, "total cells exceed configured max");
        return RC_INVALID_ARGUMENT;
    }
    if total_cells > isize::MAX as u64 {
        write_debug(debug_buffer, debug_buffer_len, "cells exceed host limits");
        return RC_OVERFLOW;
    }

    let expected_out_len = (n_rows as u64).checked_mul(HASH_SIZE as u64);
    let Some(expected_out_len) = expected_out_len else {
        write_debug(debug_buffer, debug_buffer_len, "output len overflow");
        return RC_OVERFLOW;
    };
    if expected_out_len != out_hashes_len as u64 {
        write_debug(
            debug_buffer,
            debug_buffer_len,
            "out_hashes_len mismatch",
        );
        return RC_INVALID_ARGUMENT;
    }

    let columns_bytes_len = ((total_cells) as usize).checked_mul(mem::size_of::<u32>());
    let Some(columns_bytes_len) = columns_bytes_len else {
        write_debug(debug_buffer, debug_buffer_len, "columns byte-size overflow");
        return RC_OVERFLOW;
    };

    let out_range = pointer_range(out_hashes as *const u8, expected_out_len as usize);
    let Some(out_range) = out_range else {
        write_debug(debug_buffer, debug_buffer_len, "output pointer range overflow");
        return RC_OVERFLOW;
    };

    let columns_range = pointer_range(columns_flat as *const u8, columns_bytes_len);
    let Some(columns_range) = columns_range else {
        write_debug(debug_buffer, debug_buffer_len, "columns pointer range overflow");
        return RC_OVERFLOW;
    };

    if ranges_overlap(out_range, columns_range) {
        write_debug(
            debug_buffer,
            debug_buffer_len,
            "out_hashes overlaps input buffers",
        );
        return RC_INVALID_ARGUMENT;
    }

    let mut prev_range = None;
    let prev_layer: Option<&[u8]> = if prev_layer_len == 0 {
        None
    } else {
        if prev_layer_bytes.is_null() {
            write_debug(debug_buffer, debug_buffer_len, "null prev pointer");
            return RC_NULL_POINTER;
        }
        let expected_prev_len = (n_rows as u64).checked_mul((HASH_SIZE * 2) as u64);
        let Some(expected_prev_len) = expected_prev_len else {
            write_debug(debug_buffer, debug_buffer_len, "prev len overflow");
            return RC_OVERFLOW;
        };
        if expected_prev_len != prev_layer_len as u64 {
            write_debug(
                debug_buffer,
                debug_buffer_len,
                "prev_layer_len mismatch",
            );
            return RC_INVALID_ARGUMENT;
        }
        let this_prev_range = pointer_range(prev_layer_bytes, prev_layer_len as usize);
        let Some(this_prev_range) = this_prev_range else {
            write_debug(debug_buffer, debug_buffer_len, "prev pointer range overflow");
            return RC_OVERFLOW;
        };
        prev_range = Some(this_prev_range);
        Some(slice::from_raw_parts(
            prev_layer_bytes,
            prev_layer_len as usize,
        ))
    };

    if let Some(prev_range) = prev_range {
        if ranges_overlap(out_range, prev_range) {
            write_debug(
                debug_buffer,
                debug_buffer_len,
                "out_hashes overlaps input buffers",
            );
            return RC_INVALID_ARGUMENT;
        }
    }

    if !debug_buffer.is_null() && debug_buffer_len > 0 {
        let debug_range = pointer_range(debug_buffer as *const u8, debug_buffer_len as usize);
        let Some(debug_range) = debug_range else {
            return RC_OVERFLOW;
        };
        if ranges_overlap(out_range, debug_range) {
            write_debug(
                debug_buffer,
                debug_buffer_len,
                "out_hashes overlaps debug buffer",
            );
            return RC_INVALID_ARGUMENT;
        }
    }

    let columns = slice::from_raw_parts(columns_flat, total_cells as usize);

    let mut col_indices = Vec::with_capacity(n_columns as usize);
    for col in 0..n_columns {
        col_indices.push(col.to_le_bytes());
    }
    let log_size_bytes = log_size.to_le_bytes();
    let hash_state_prefix = build_hash_state_prefix(&log_size_bytes);

    let n_rows_usize = n_rows as usize;
    let n_columns_usize = n_columns as usize;
    let use_parallel = should_use_parallel(total_cells, n_rows);
    {
        let out = slice::from_raw_parts_mut(out_hashes, expected_out_len as usize);
        if use_parallel {
            out.par_chunks_mut(HASH_SIZE)
                .with_min_len(PARALLEL_MIN_CHUNK_ROWS)
                .enumerate()
                .for_each(|(row, out_row)| {
                    hash_row(
                        row,
                        out_row,
                        &hash_state_prefix,
                        prev_layer,
                        &col_indices,
                        n_columns_usize,
                        n_rows_usize,
                        columns,
                    );
                });
        } else {
            out.chunks_mut(HASH_SIZE).enumerate().for_each(|(row, out_row)| {
                hash_row(
                    row,
                    out_row,
                    &hash_state_prefix,
                    prev_layer,
                    &col_indices,
                    n_columns_usize,
                    n_rows_usize,
                    columns,
                );
            });
        }
    }

    if debug_level > 0 {
        let mode = if use_parallel { "parallel" } else { "serial" };
        let msg = format!("ok rows={} cols={} mode={}", n_rows, n_columns, mode);
        write_debug(debug_buffer, debug_buffer_len, &msg);
    } else {
        write_debug(debug_buffer, debug_buffer_len, "");
    }
    RC_OK
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn hash_row(
    row: usize,
    out_row: &mut [u8],
    hash_state_prefix: &blake2s_simd::State,
    prev_layer: Option<&[u8]>,
    col_indices: &[[u8; 4]],
    n_columns_usize: usize,
    n_rows_usize: usize,
    columns: &[u32],
) {
    let mut state = hash_state_prefix.clone();
    state.update(&(row as u32).to_le_bytes());

    if let Some(prev_layer) = prev_layer {
        let base = row * HASH_SIZE * 2;
        state.update(&prev_layer[base..base + HASH_SIZE]);
        state.update(&prev_layer[base + HASH_SIZE..base + HASH_SIZE * 2]);
    }

    for (col, col_bytes) in col_indices.iter().enumerate().take(n_columns_usize) {
        state.update(col_bytes);
        let idx = col * n_rows_usize + row;
        state.update(&columns[idx].to_le_bytes());
    }

    let digest = state.finalize();
    out_row.copy_from_slice(digest.as_bytes());
}

fn build_hash_state_prefix(log_size_bytes: &[u8; 4]) -> blake2s_simd::State {
    let mut params = blake2s_simd::Params::new();
    params.hash_length(HASH_SIZE);
    params.personal(PERSONALIZATION);

    let mut state = params.to_state();
    state.update(DOMAIN_TAG);
    state.update(log_size_bytes);
    state
}

fn should_use_parallel(total_cells: u64, n_rows: u32) -> bool {
    should_use_parallel_with_threads(total_cells, n_rows, rayon::current_num_threads())
}

fn should_use_parallel_with_threads(total_cells: u64, n_rows: u32, n_threads: usize) -> bool {
    total_cells >= PARALLEL_MIN_CELLS && n_rows >= PARALLEL_MIN_ROWS && n_threads > 1
}

fn should_use_parallel_m31(len: usize) -> bool {
    should_use_parallel_m31_with_threads(len, rayon::current_num_threads())
}

fn should_use_parallel_m31_with_threads(len: usize, n_threads: usize) -> bool {
    len >= PARALLEL_MIN_M31_LEN && n_threads > 1
}

#[inline]
fn m31_axpy_scalar(alpha: u32, beta: u32, a: u32, b: u32, c: u32) -> u32 {
    // Keep intermediates in the M31 field using fast Mersenne reductions.
    // This avoids generic `%` division in the hot path.
    let t0 = m31_mul(alpha, a);
    let t1 = m31_mul(beta, b);
    let t2 = m31_add(t0, t1);
    m31_add(t2, c)
}

#[inline]
fn m31_mul(lhs: u32, rhs: u32) -> u32 {
    m31_reduce_u64((lhs as u64) * (rhs as u64))
}

#[inline]
fn m31_add(lhs: u32, rhs: u32) -> u32 {
    let sum = lhs + rhs;
    if sum >= M31_PRIME {
        sum - M31_PRIME
    } else {
        sum
    }
}

#[inline]
fn m31_reduce_u64(val: u64) -> u32 {
    // Same reduction identity used for M31 in STWO: valid for products in [0, P^2).
    // (lhs*rhs) always satisfies this precondition for field elements.
    (((((val >> 31) + val + 1) >> 31) + val) & (M31_PRIME as u64)) as u32
}

/// Runs one deterministic malformed-input stress case for the C ABI entrypoint.
///
/// This helper is intended for `cargo-fuzz` and deterministic regression tests.
/// It should never panic; return values are expected to be one of the ABI status codes.
#[doc(hidden)]
pub fn fuzz_abi_entrypoint(data: &[u8]) -> i32 {
    let mut cursor = FuzzCursor::new(data);

    let flags = cursor.next_u32();
    let log_size = cursor.next_u32() % 26;

    let mut n_columns = cursor.next_u32() % (MAX_COLUMNS + 128);
    let mut n_rows = match cursor.next_u32() & 0b11 {
        0 => 0,
        1 => 1u32.checked_shl(log_size).unwrap_or(0),
        2 => cursor.next_u32() % (1 << 18),
        _ => cursor.next_u32(),
    };
    if (flags & FLAG_FORCE_VALID_SHAPE) != 0 {
        n_rows = 1u32.checked_shl(log_size).unwrap_or(0);
        if n_columns == 0 {
            n_columns = 1;
        }
    }

    let total_cells = (n_rows as u64).saturating_mul(n_columns as u64);
    let alloc_cells = (total_cells.min(FUZZ_MAX_CELLS as u64) as usize).max(1);

    let mut columns_aligned = vec![0u32; alloc_cells];
    cursor.fill_u32_words(&mut columns_aligned);

    let misaligned_len = alloc_cells
        .saturating_mul(mem::size_of::<u32>())
        .saturating_add(8)
        .max(8);
    let mut columns_misaligned = vec![0u8; misaligned_len];
    cursor.fill_bytes(&mut columns_misaligned);

    let prev_layer_len = cursor.next_u32().min(FUZZ_MAX_BUFFER_BYTES);
    let mut prev_storage = vec![0u8; (prev_layer_len as usize).max(1)];
    cursor.fill_bytes(&mut prev_storage);

    let mut out_hashes_len = cursor.next_u32().min(FUZZ_MAX_BUFFER_BYTES);
    let mut out_storage = vec![0u8; (out_hashes_len as usize).max(1)];
    cursor.fill_bytes(&mut out_storage);

    let mut debug_buffer_len = cursor.next_u32();
    if (flags & FLAG_OVERSIZE_DEBUG_LEN) != 0 {
        debug_buffer_len = MAX_DEBUG_BUFFER_LEN + 1;
    } else {
        debug_buffer_len = debug_buffer_len.min(MAX_DEBUG_BUFFER_LEN);
    }
    let debug_storage_len = if debug_buffer_len > MAX_DEBUG_BUFFER_LEN {
        1
    } else {
        (debug_buffer_len as usize).max(1)
    };
    let mut debug_storage = vec![0u8; debug_storage_len];
    cursor.fill_bytes(&mut debug_storage);

    let mut columns_ptr = if (flags & FLAG_NULL_COLUMNS) != 0 {
        std::ptr::null()
    } else if (flags & FLAG_MISALIGN_COLUMNS) != 0 {
        columns_misaligned.as_ptr().wrapping_add(1) as *const u32
    } else {
        columns_aligned.as_ptr()
    };

    let mut prev_ptr = if (flags & FLAG_NULL_PREV) != 0 {
        std::ptr::null()
    } else {
        prev_storage.as_ptr()
    };

    let mut out_ptr = if (flags & FLAG_NULL_OUT) != 0 {
        std::ptr::null_mut()
    } else {
        out_storage.as_mut_ptr()
    };

    let mut debug_ptr = if (flags & FLAG_NULL_DEBUG) != 0 {
        std::ptr::null_mut()
    } else {
        debug_storage.as_mut_ptr() as *mut c_char
    };

    if (flags & FLAG_ALIAS_OUT_COLUMNS) != 0 && !out_ptr.is_null() && !columns_ptr.is_null() {
        let columns_bytes_capacity = if (flags & FLAG_MISALIGN_COLUMNS) != 0 {
            columns_misaligned.len().saturating_sub(1)
        } else {
            columns_aligned
                .len()
                .saturating_mul(mem::size_of::<u32>())
        };
        out_hashes_len = out_hashes_len.min(columns_bytes_capacity as u32);
        out_ptr = columns_ptr as *mut u8;
    } else if (flags & FLAG_ALIAS_OUT_PREV) != 0 && !out_ptr.is_null() && !prev_ptr.is_null() {
        out_hashes_len = out_hashes_len.min(prev_storage.len() as u32);
        out_ptr = prev_ptr as *mut u8;
    } else if (flags & FLAG_ALIAS_OUT_DEBUG) != 0 && !out_ptr.is_null() && !debug_ptr.is_null() {
        out_hashes_len = out_hashes_len.min(debug_storage.len() as u32);
        out_ptr = debug_ptr as *mut u8;
    }

    if (flags & FLAG_NULL_COLUMNS) != 0 {
        columns_ptr = std::ptr::null();
    }
    if (flags & FLAG_NULL_PREV) != 0 {
        prev_ptr = std::ptr::null();
    }
    if (flags & FLAG_NULL_OUT) != 0 {
        out_ptr = std::ptr::null_mut();
    }
    if (flags & FLAG_NULL_DEBUG) != 0 {
        debug_ptr = std::ptr::null_mut();
    }

    let rc = unsafe {
        mojo_blake2s_commit_layer(
            log_size,
            prev_ptr,
            prev_layer_len,
            columns_ptr,
            n_columns,
            n_rows,
            out_ptr,
            out_hashes_len,
            cursor.next_u32() % 4,
            debug_ptr,
            debug_buffer_len,
        )
    };

    debug_assert!(matches!(
        rc,
        RC_OK | RC_NULL_POINTER | RC_INVALID_ARGUMENT | RC_OVERFLOW | RC_PANIC
    ));
    rc
}

struct FuzzCursor<'a> {
    data: &'a [u8],
    idx: usize,
}

impl<'a> FuzzCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, idx: 0 }
    }

    fn next_u8(&mut self) -> u8 {
        if self.data.is_empty() {
            return 0;
        }
        let byte = self.data[self.idx % self.data.len()];
        self.idx = self.idx.wrapping_add(1);
        byte
    }

    fn next_u32(&mut self) -> u32 {
        let b0 = self.next_u8() as u32;
        let b1 = self.next_u8() as u32;
        let b2 = self.next_u8() as u32;
        let b3 = self.next_u8() as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    fn fill_bytes(&mut self, out: &mut [u8]) {
        for byte in out {
            *byte = self.next_u8();
        }
    }

    fn fill_u32_words(&mut self, out: &mut [u32]) {
        for word in out {
            *word = self.next_u32();
        }
    }
}

fn write_debug(buffer: *mut c_char, buffer_len: u32, msg: &str) {
    if buffer.is_null() || buffer_len == 0 {
        return;
    }
    unsafe {
        let raw = slice::from_raw_parts_mut(buffer as *mut u8, buffer_len as usize);
        raw.fill(0);
        let max_copy = raw.len().saturating_sub(1);
        let bytes = msg.as_bytes();
        let n = bytes.len().min(max_copy);
        raw[..n].copy_from_slice(&bytes[..n]);
    }
}

fn pointer_range(ptr: *const u8, len: usize) -> Option<(usize, usize)> {
    let start = ptr as usize;
    let end = start.checked_add(len)?;
    Some((start, end))
}

fn ranges_overlap(lhs: (usize, usize), rhs: (usize, usize)) -> bool {
    lhs.0 < rhs.1 && rhs.0 < lhs.1
}

#[cfg(test)]
mod tests {
    use super::{
        fuzz_abi_entrypoint, m31_axpy_scalar, m31_mul, mojo_kernel_abi_version, mojo_m31_axpy,
        should_use_parallel_m31_with_threads, should_use_parallel_with_threads, M31_PRIME,
        MOJO_KERNEL_ABI_VERSION, RC_INVALID_ARGUMENT, RC_NULL_POINTER, RC_OK, RC_OVERFLOW,
        RC_PANIC,
    };

    #[test]
    fn fuzz_smoke_regression_inputs_do_not_panic() {
        let corpus: [&[u8]; 8] = [
            b"",
            b"\x00",
            b"\xff",
            b"\x00\x00\x00\x00\x00\x00\x00\x00",
            b"\x01\x02\x03\x04\x05\x06\x07\x08\x09",
            b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff",
            b"starknet-mojo-fuzz",
            b"\x10\x00\x00\x00\x01\x00\x00\x00\x20\x00\x00\x00\x20\x00\x00\x00",
        ];
        for input in corpus {
            let rc = fuzz_abi_entrypoint(input);
            assert!(matches!(
                rc,
                RC_OK | RC_NULL_POINTER | RC_INVALID_ARGUMENT | RC_OVERFLOW | RC_PANIC
            ));
        }
    }

    #[test]
    fn fuzz_deterministic_stream_no_unexpected_status_codes() {
        let mut state = 0xA5A5_5A5A_F0F0_0F0Fu64;
        let mut buf = [0u8; 256];

        for _ in 0..256 {
            let len = ((next_u64(&mut state) as usize) % buf.len()).max(1);
            for byte in buf.iter_mut().take(len) {
                *byte = (next_u64(&mut state) & 0xFF) as u8;
            }
            let rc = fuzz_abi_entrypoint(&buf[..len]);
            assert!(matches!(
                rc,
                RC_OK | RC_NULL_POINTER | RC_INVALID_ARGUMENT | RC_OVERFLOW | RC_PANIC
            ));
        }
    }

    #[test]
    fn strategy_prefers_serial_for_small_workloads() {
        assert!(!should_use_parallel_with_threads(1024, 128, 4));
        assert!(!should_use_parallel_with_threads(16_383, 255, 4));
        assert!(!should_use_parallel_with_threads(32_767, 512, 4));
        assert!(!should_use_parallel_with_threads(32_768, 512, 1));
    }

    #[test]
    fn strategy_enables_parallel_for_large_workloads() {
        assert!(should_use_parallel_with_threads(32_768, 512, 2));
        assert!(should_use_parallel_with_threads(200_000, 1024, 8));
    }

    #[test]
    fn m31_strategy_prefers_serial_for_small_workloads() {
        assert!(!should_use_parallel_m31_with_threads(1024, 4));
        assert!(!should_use_parallel_m31_with_threads(16_383, 4));
        assert!(!should_use_parallel_m31_with_threads(16_384, 1));
    }

    #[test]
    fn m31_strategy_enables_parallel_for_large_workloads() {
        assert!(should_use_parallel_m31_with_threads(16_384, 2));
        assert!(should_use_parallel_m31_with_threads(32_768, 8));
    }

    #[test]
    fn m31_scalar_matches_expected_mod_arithmetic() {
        let out = m31_axpy_scalar(7, 9, 11, 13, 17);
        let expected = ((7u64 * 11u64 + 9u64 * 13u64 + 17u64) % M31_PRIME as u64) as u32;
        assert_eq!(out, expected);
    }

    #[test]
    fn m31_mul_matches_expected_mod_arithmetic() {
        let lhs = M31_PRIME - 2;
        let rhs = M31_PRIME - 3;
        let out = m31_mul(lhs, rhs);
        let expected = ((lhs as u64 * rhs as u64) % M31_PRIME as u64) as u32;
        assert_eq!(out, expected);
    }

    #[test]
    fn m31_scalar_matches_expected_mod_arithmetic_wide_cases() {
        let mut state = 0xBEEF_CAFE_1234_5678u64;
        for _ in 0..10_000 {
            let a = (next_u64(&mut state) as u32) % M31_PRIME;
            let b = (next_u64(&mut state) as u32) % M31_PRIME;
            let c = (next_u64(&mut state) as u32) % M31_PRIME;
            let alpha = (next_u64(&mut state) as u32) % M31_PRIME;
            let beta = (next_u64(&mut state) as u32) % M31_PRIME;
            let out = m31_axpy_scalar(alpha, beta, a, b, c);
            let expected =
                ((alpha as u64 * a as u64 + beta as u64 * b as u64 + c as u64) % M31_PRIME as u64)
                    as u32;
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn m31_abi_rejects_out_of_field_input() {
        let a = [M31_PRIME];
        let b = [2u32];
        let c = [3u32];
        let mut out = [0u32; 1];
        let mut debug = [0i8; 128];
        let rc = unsafe {
            mojo_m31_axpy(
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                1,
                7,
                11,
                out.as_mut_ptr(),
                1,
                1,
                debug.as_mut_ptr(),
                debug.len() as u32,
            )
        };
        assert_eq!(rc, RC_INVALID_ARGUMENT);
    }

    #[test]
    fn abi_version_matches_expected() {
        assert_eq!(mojo_kernel_abi_version(), MOJO_KERNEL_ABI_VERSION);
    }

    fn next_u64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }
}
