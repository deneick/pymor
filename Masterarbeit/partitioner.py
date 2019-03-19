from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.grids.tria import TriaGrid

def partition_any_grid(grid, num_intervals=(2,2), codim=2, dmask = None):
    subdomains = [[[] for _ in range(num_intervals[1])] for _ in range(num_intervals[0])]
    x0_edges = [[[] for _ in range(num_intervals[1] - 1)] for _ in range(num_intervals[0])]
    x1_edges = [[[] for _ in range(num_intervals[1])] for _ in range(num_intervals[0] - 1)]
    vertices = [[[] for _ in range(num_intervals[1] - 1)] for _ in range(num_intervals[0] - 1)]
    
    x0_increment = float(grid.x0_width) / num_intervals[0]
    x1_increment = float(grid.x1_width) / num_intervals[1]
    
    def on_x0_boundary(coord):
        x0 = coord[0]
        x0_in_increments = (x0 - grid.x0_range[0]) / x0_increment
        if np.isclose(np.round(x0_in_increments), x0_in_increments) and x0_in_increments > 0.5 and x0_in_increments < num_intervals[0] - 0.5:
            return True
        else:
            return False

    def on_x1_boundary(coord):
        x1 = coord[1]
        x1_in_increments = (x1 - grid.x1_range[0]) / x1_increment
        if np.isclose(np.round(x1_in_increments), x1_in_increments) and x1_in_increments > 0.5 and x1_in_increments < num_intervals[1] - 0.5:
            return True
        else:
            return False
            
    for dofnr, coord in enumerate(grid.centers(codim)):
#	if (not (dmask is None)) and dmask[dofnr]:
#		continue
        if on_x0_boundary(coord):
            if on_x1_boundary(coord):
                vertices[int(np.round((coord[0] - grid.x0_range[0]) / x0_increment) - 1)][int(np.round((coord[1] - grid.x1_range[0]) / x1_increment) - 1)].append(dofnr)
            else:
                x1_edges[int(np.round((coord[0] - grid.x0_range[0]) / x0_increment) - 1)][min(int(np.floor((coord[1] - grid.x1_range[0]) / x1_increment)), num_intervals[1] - 1)].append(dofnr)
        elif on_x1_boundary(coord):
            x0_edges[min(int(np.floor((coord[0] - grid.x0_range[0]) / x0_increment)), num_intervals[0] - 1)][int(np.round((coord[1] - grid.x1_range[0]) / x1_increment) - 1)].append(dofnr)
        else:
            subdomains[min(int(np.floor((coord[0] - grid.x0_range[0]) / x0_increment)), num_intervals[0] - 1)][min(int(np.floor((coord[1] - grid.x1_range[0]) / x1_increment)), num_intervals[1] - 1)].append(dofnr)
            
    def to_np(l):
        return [[np.array(a, dtype=np.dtype('int32')) for a in ab] for ab in l]
        
    return to_np(subdomains), (to_np(x0_edges), to_np(x1_edges)), to_np(vertices)

def partition_tria_grid(grid, num_intervals=(2, 2)):
    assert isinstance(grid, TriaGrid)
    assert not grid.identify_left_right and not grid.identify_bottom_top
    assert grid.num_intervals[0] % num_intervals[0] == 0
    assert grid.num_intervals[1] % num_intervals[1] == 0

    gni = grid.num_intervals
    ni = num_intervals
    gni_per_i = (gni[0] / ni[0], gni[1] / ni[1])

    vertex_numbers = (np.arange((grid.num_intervals[0] + 1) * (grid.num_intervals[1] + 1))
                      .reshape((grid.num_intervals[1] + 1, grid.num_intervals[0] + 1)).T)

    center_vertex_numbers = (np.arange(grid.num_intervals[0] * grid.num_intervals[1])
                             .reshape(grid.num_intervals[1], grid.num_intervals[0]).T) + vertex_numbers.size

    subdomains = np.array([[vertex_numbers[i0 * gni_per_i[0] + 1:(i0 + 1) * gni_per_i[0],
                                           i1 * gni_per_i[1] + 1:(i1 + 1) * gni_per_i[1]].ravel()
                            for i1 in range(ni[1])]
                           for i0 in range(ni[0])])

    subdomain_centers = np.array([[center_vertex_numbers[i0 * gni_per_i[0]:(i0 + 1) * gni_per_i[0],
                                                         i1 * gni_per_i[1]:(i1 + 1) * gni_per_i[1]].ravel()
                                  for i1 in range(ni[1])]
                                 for i0 in range(ni[0])])

    subdomains = np.concatenate((subdomains, subdomain_centers), axis=2)
    del subdomain_centers

    x0_edges = np.array([[vertex_numbers[i0 * gni_per_i[0] + 1:(i0 + 1) * gni_per_i[0],
                                         i1 * gni_per_i[1]].ravel()
                          for i1 in range(ni[1] + 1)]
                         for i0 in range(ni[0])])

    x1_edges = np.array([[vertex_numbers[i0 * gni_per_i[0],
                                         i1 * gni_per_i[1] + 1:(i1 + 1) * gni_per_i[1]].ravel()
                          for i1 in range(ni[1])]
                         for i0 in range(ni[0] + 1)])

    vertices = np.array([[[vertex_numbers[i0 * gni_per_i[0],
                                          i1 * gni_per_i[1]]]
                          for i1 in range(ni[1] + 1)]
                         for i0 in range(ni[0] + 1)])

    return subdomains, (x0_edges, x1_edges), vertices


def merge_domain_boundaries(subdomains, edges, vertices):

    subdomains = [[s for s in sd] for sd in subdomains]
    subdomains[0] = [np.concatenate((s, e)) for s, e in zip(subdomains[0], edges[1][0])]
    subdomains[-1] = [np.concatenate((s, e)) for s, e in zip(subdomains[-1], edges[1][-1])]
    for i in range(len(subdomains)):
        subdomains[i][0] = np.concatenate((subdomains[i][0], edges[0][i][0]))
        subdomains[i][-1] = np.concatenate((subdomains[i][-1], edges[0][i][-1]))
    for i, j in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
        subdomains[i][j] = np.concatenate((subdomains[i][j], vertices[i][j]))

    x0_edges, x1_edges = edges[0][:, 1:-1], edges[1][1:-1, :]
    x0_edges = [[e for e in ed] for ed in x0_edges]
    x1_edges = [[e for e in ed] for ed in x1_edges]
    x0_edges[0] = [np.concatenate((e, v)) for e, v in zip(x0_edges[0], vertices[0, 1:-1])]
    x0_edges[-1] = [np.concatenate((e, v)) for e, v in zip(x0_edges[-1], vertices[-1, 1:-1])]
    for i in range(len(x1_edges)):
        x1_edges[i][0] = np.concatenate((x1_edges[i][0], vertices[i + 1, 0]))
        x1_edges[i][-1] = np.concatenate((x1_edges[i][-1], vertices[i + 1, -1]))

    vertices = [[v for v in vt] for vt in vertices[1:-1, 1:-1]]

    for thing in [subdomains, x0_edges, x1_edges, vertices]:
        for ab in thing:
            for a in ab:
                a.sort()
        
    return subdomains, (x0_edges, x1_edges), vertices


def build_subspaces(subdomains, edges, vertices):

    dtype = {'names': ['id', 'codim', 'patch', 'cpatch', 'hpatch', 'env', 'cenv', 'henv', 'dofs', 'xenv', 'cxenv'],
             'formats': ['int32', 'int8', 'object', 'object', 'object', 'object', 'object', 'object', 'object', 'object', 'object']}

    sd_x0 = len(subdomains)
    sd_x1 = len(subdomains[0])
    sd = np.zeros((sd_x0, sd_x1), dtype=dtype)

    sd['dofs'] = subdomains
    sd['id'] = np.arange(sd_x0 * sd_x1).reshape((sd_x0, sd_x1))
    sd['codim'] = 0

    x0_ed_x0 = len(edges[0])
    x0_ed_x1 = len(edges[0][0])
    x0_ed = np.zeros((x0_ed_x0, x0_ed_x1), dtype=dtype)

    x0_ed['dofs'] = edges[0]
    x0_ed['id'] = np.arange(x0_ed_x0 * x0_ed_x1).reshape((x0_ed_x0, x0_ed_x1)) + sd.size
    x0_ed['codim'] = 1

    x1_ed_x0 = len(edges[1])
    x1_ed_x1 = len(edges[1][0])
    x1_ed = np.zeros((x1_ed_x0, x1_ed_x1), dtype=dtype)

    x1_ed['dofs'] = edges[1]
    x1_ed['id'] = np.arange(x1_ed_x0 * x1_ed_x1).reshape((x1_ed_x0, x1_ed_x1)) + (sd.size + x0_ed.size)
    x1_ed['codim'] = 1

    vt_x0 = len(vertices)
    vt_x1 = len(vertices[1])
    vt = np.zeros((vt_x0, vt_x1), dtype=dtype)
    vt['dofs'] = vertices
    vt['id'] = np.arange(vt_x0 * vt_x1).reshape((vt_x0, vt_x1)) + (sd.size + x0_ed.size + x1_ed.size)
    vt['codim'] = 2

    for i, s in np.ndenumerate(sd):
        s['patch'] = (s['id'],)
        cpatch = []
        if i[0] > 0:
            cpatch.append(x1_ed['id'][i[0] - 1, i[1]])
        if i[0] + 1 < sd_x0:
            cpatch.append(x1_ed['id'][i])
        if i[1] > 0:
            cpatch.append(x0_ed['id'][i[0], i[1] - 1])
        if i[1] + 1 < sd_x1:
            cpatch.append(x0_ed['id'][i])

        if i[0] > 0 and i[1] > 0:
            cpatch.append(vt['id'][i[0] - 1, i[1] - 1])
        if i[0] > 0 and i[1] + 1 < sd_x1:
            cpatch.append(vt['id'][i[0] - 1, i[1]])
        if i[0] + 1 < sd_x0 and i[1] > 0:
            cpatch.append(vt['id'][i[0], i[1] - 1])
        if i[0] + 1 < sd_x0 and i[1] + 1 < sd_x1:
            cpatch.append(vt['id'][i[0], i[1]])

        s['cpatch'] = tuple(cpatch)
        s['hpatch'] = tuple()

    for i, e in np.ndenumerate(x0_ed):
        patch = [e['id'], sd['id'][i[0], i[1]], sd['id'][i[0], i[1] + 1]]
        e['patch'] = tuple(patch)
        e['hpatch'] = tuple([x for x in patch if not x == e['id']])

        if i[0] > 0:
            patch.extend([sd['id'][i[0] - 1, i[1]], sd['id'][i[0] - 1, i[1] + 1],
                          x0_ed['id'][i[0] - 1, i[1]],
                          x1_ed['id'][i[0] - 1, i[1]], x1_ed['id'][i[0] - 1, i[1] + 1],
                          vt['id'][i[0] - 1, i[1]]])
        if i[0] + 1 < sd_x0:
            patch.extend([sd['id'][i[0] + 1, i[1]], sd['id'][i[0] + 1, i[1] + 1],
                          x0_ed['id'][i[0] + 1, i[1]],
                          x1_ed['id'][i[0], i[1]], x1_ed['id'][i[0], i[1] + 1],
                          vt['id'][i[0], i[1]]])
        e['env'] = tuple(patch)
        e['henv'] = tuple([x for x in patch if not x == e['id']])

    for i, e in np.ndenumerate(x1_ed):
        patch = [e['id'], sd['id'][i[0], i[1]], sd['id'][i[0] + 1, i[1]]]
        e['patch'] = tuple(patch)
        e['hpatch'] = tuple([x for x in patch if not x == e['id']])

        if i[1] > 0:
            patch.extend([sd['id'][i[0], i[1] - 1], sd['id'][i[0] + 1, i[1] - 1],
                          x1_ed['id'][i[0], i[1] - 1],
                          x0_ed['id'][i[0], i[1] - 1], x0_ed['id'][i[0] + 1, i[1] - 1],
                          vt['id'][i[0], i[1] - 1]])
        if i[1] + 1 < sd_x1:
            patch.extend([sd['id'][i[0], i[1] + 1], sd['id'][i[0] + 1, i[1] + 1],
                          x1_ed['id'][i[0], i[1] + 1],
                          x0_ed['id'][i[0], i[1]], x0_ed['id'][i[0] + 1, i[1]],
                          vt['id'][i[0], i[1]]])
        e['env'] = tuple(patch)
        e['henv'] = tuple([x for x in patch if not x == e['id']])

    for i, v in np.ndenumerate(vt):
        v['patch'] = tuple(np.concatenate(([v['id']],
                                           sd['id'][i[0]:i[0] + 2, i[1]:i[1] + 2].ravel(),
                                           x0_ed['id'][i[0]:i[0] + 2, i[1]],
                                           x1_ed['id'][i[0], i[1]:i[1] + 2])))
        v['env'] = v['patch']
        v['hpatch'] = tuple([x for x in v['patch'] if not x == v['id']])
        v['henv'] = tuple([x for x in v['env'] if not x == v['id']])

    subspaces = np.concatenate((sd.ravel(), x0_ed.ravel(), x1_ed.ravel(), vt.ravel()))
    subspaces_per_codim = (sd['id'].ravel(),
                           np.concatenate((x0_ed['id'].ravel(), x1_ed['id'].ravel())),
                           vt['id'].ravel())

    for s in subspaces:
        if s['codim'] == 0:
            s['env'] = tuple(set(i
                                 for k in s['cpatch'] if subspaces[k]['codim'] == 1
                                 for i in subspaces[k]['env']))
            s['henv'] = tuple([x for x in s['env'] if not x == s['id']])
        else:
            s['cpatch'] = tuple(set(i
                                    for k in s['patch'] if subspaces[k]['codim'] == 0
                                    for i in subspaces[k]['cpatch'] if i not in s['patch']))
        s['cenv'] = tuple(set(i
                              for k in s['env'] if subspaces[k]['codim'] == 0
                              for i in subspaces[k]['cpatch'] if i not in s['env']))

        s['xenv'] = tuple(set(i
                              for k in s['cenv'] if subspaces[k]['codim'] == 1
                              for i in subspaces[k]['env'])
                          | set(s['env']))
        
        s['cxenv'] = tuple(set(i
                               for k in s['xenv'] if subspaces[k]['codim'] == 0
                               for i in subspaces[k]['cpatch'] if i not in s['xenv']))

    # sort all:
    for x in subspaces:
        for spacename in ['patch', 'cpatch', 'hpatch', 'env', 'cenv', 'henv', 'xenv', 'cxenv']:
            s[spacename] = tuple(sorted(s[spacename]))

    return subspaces, subspaces_per_codim
