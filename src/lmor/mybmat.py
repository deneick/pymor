import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.sputils import upcast


def mybmat(blocks, format=None, dtype=None):
    # bugfix-Zeile
    if len(blocks) > 0:
        numcols = [None for _ in blocks[0]]
        for row in blocks:
            for colnr, col in enumerate(row):
                if col is not None:
                    numcols[colnr] = col.shape[1]

        assert not (None in numcols)
        blocks.append([np.zeros(shape=(0, i)) for i in numcols])

    blocks = np.asarray(blocks, dtype='object')
    if np.ndim(blocks) != 2:
        raise ValueError('blocks must have rank 2')

    M, N = blocks.shape

    block_mask = np.zeros(blocks.shape, dtype=np.bool)
    brow_lengths = np.zeros(blocks.shape[0], dtype=np.intc)
    bcol_lengths = np.zeros(blocks.shape[1], dtype=np.intc)

    # convert everything to COO format
    for i in range(M):
        for j in range(N):
            if blocks[i, j] is not None:
                A = coo_matrix(blocks[i, j])
                blocks[i, j] = A
                block_mask[i, j] = True

                if brow_lengths[i] == 0:
                    brow_lengths[i] = A.shape[0]
                else:
                    if brow_lengths[i] != A.shape[0]:
                        raise ValueError('blocks[%d,:] has incompatible row dimensions' % i)

                if bcol_lengths[j] == 0:
                    bcol_lengths[j] = A.shape[1]
                else:
                    if bcol_lengths[j] != A.shape[1]:
                        raise ValueError('blocks[:,%d] has incompatible column dimensions' % j)

    nnz = sum([A.nnz for A in blocks[block_mask]])
    if dtype is None:
        dtype = upcast(*tuple([A.dtype for A in blocks[block_mask]]))

    row_offsets = np.concatenate(([0], np.cumsum(brow_lengths)))
    col_offsets = np.concatenate(([0], np.cumsum(bcol_lengths)))

    data = np.empty(nnz, dtype=dtype)
    row = np.empty(nnz, dtype=np.intc)
    col = np.empty(nnz, dtype=np.intc)

    nnz = 0
    for i in range(M):
        for j in range(N):
            if blocks[i, j] is not None:
                A = blocks[i, j]
                data[nnz:nnz + A.nnz] = A.data
                row[nnz:nnz + A.nnz] = A.row
                col[nnz:nnz + A.nnz] = A.col

                row[nnz:nnz + A.nnz] += row_offsets[i]
                col[nnz:nnz + A.nnz] += col_offsets[j]

                nnz += A.nnz

    shape = (np.sum(brow_lengths), np.sum(bcol_lengths))
    return coo_matrix((data, (row, col)), shape=shape).asformat(format)
