"""
Author: Liam Murray, murrayla@student.unimelb.edu.au
Descrption: cmiss setup funtions for finite element implementation of 
                cylinder inflation
"""

import numpy as np
from opencmiss.iron import iron
import meshio

# Runtime required parameters
DIM = 3
N_N_EL = 27
QUAD_ORDER = 4
X, Y, Z, P = (1, 2, 3, 4)
PRESSURE_TEST = True
LOADSTEPS = 10
INNER_RAD = 0.375
RUNTIME_PATH = "/home/jovyan/work/docker-iron/test_files/inflation_test/runtime_files/"

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Coordinate System:
#   Simply defines the coordinate system that is used for 
#       solving the finite element problem (i.e. 3D).
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def coordinate_setup(coord_n, dim):
    coord_sys = iron.CoordinateSystem()
    # + 1
    coord_sys.CreateStart(coord_n)
    coord_sys.DimensionSet(dim)
    # - 1
    coord_sys.CreateFinish()
    print('+==+ ^\_/^ COORDINATES COMPLETE')
    return coord_sys

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Basis:
#   Denotes the interpolation of the problem geometry. 
#       The type of basis here is LAGRANGE_HERMITE_TP which
#       is describing the form of the interpolation functions. This 
#       inidicates the sort of continuity expected between each node.  
#       The Interpolation Scheme is denoted here as QUADRATIC_LAGRANGE, 
#       meaning there are 3 nodes along each direction i.e.,
#       -x <- n1 ---- n2 ---- n3 -> x
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def basis_setup(basis_n, xi_n):
    basis = iron.Basis()
    # + 1
    basis.CreateStart(basis_n)
    basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
    basis.NumberOfXiSet(xi_n)
    basis.InterpolationXiSet(
        [iron.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE] * xi_n
    )
    basis.QuadratureNumberOfGaussXiSet([3]*xi_n)
    # - 1
    basis.CreateFinish()
    print('+==+ ^\_/^ BASIS COMPLETE')
    return basis

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Region: 
#   The region is where a certain physics is occuring, we can always
#       specify multiple regions to couple different operations and
#       seperate them in our computations.
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def region_setup(region_n, coord_sys):
    region = iron.Region()
    # + 1
    region.CreateStart(region_n, iron.WorldRegion)
    region.CoordinateSystemSet(coord_sys)
    region.LabelSet("Region")
    # - 1
    region.CreateFinish()
    print('+==+ ^\_/^ REGION COMPLETE')
    return region

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Mesh:
#   This module is very explicitly defining the mesh for OpenCMISS,
#       it is possible to have these pre-generated within iron
#       which leverages external packages like TetGen. In the below
#       determined Node and Element values are being prescribed.
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def mesh_setup(region, n_n, mesh_n, dim, e_n, basis, e_idx, e_np_map):
    nodesList = iron.Nodes()
    # + 1
    nodesList.CreateStart(region, n_n)
    # - 1
    nodesList.CreateFinish()
    mesh = iron.Mesh()
    # + 2
    mesh.CreateStart(mesh_n, region, dim)
    mesh.NumberOfElementsSet(e_n)
    mesh.NumberOfComponentsSet(1) # =~-~=~-~= ?
    mesh_e = iron.MeshElements()
    # + 3
    mesh_e.CreateStart(mesh, 1, basis)
    for i in range(e_n):
        nodesList = list(
            map(int,e_np_map[i][:])
        )
        mesh_e.NodesSet(e_idx[i], nodesList)
    # - 3
    mesh_e.CreateFinish()
    # - 2
    mesh.CreateFinish()
    print('+==+ ^\_/^ MESH COMPLETE')
    return mesh_e, mesh

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Decomposition:
#   This step is implementing parallel processing by
#       breaking the mesh into different components for
#       calculation. The Domains are each a parallel stream
#       which is set by "comp_nodes_n"; the available 
#       quantity of nodes. The parameter "CALCULATED" indicates
#       that OpenCMISS will calculate the decomposition with
#       a graph partitioning approach.
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def decomposition_setup(mesh, decomp_n):
    comp_nodes_n = iron.ComputationalNumberOfNodesGet()
    decomp = iron.Decomposition()
    # + 1
    decomp.CreateStart(decomp_n, mesh)
    decomp.TypeSet(iron.DecompositionTypes.CALCULATED)
    decomp.NumberOfDomainsSet(comp_nodes_n)
    # - 1
    decomp.CreateFinish()
    print('+==+ ^\_/^ DECOMPOSITION COMPLETE')
    return decomp

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Geometric Field:
#   Once a decomposition into domains has been set up, these domains
#       can have parts of the geometry allocated to them. This is the
#       geometric field. Variables in this field and others can have
#       labels associated with them. Field values are by default 0
#       so after creating the field we can then allocate them values. 
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def geometric_setup(geo_field_n, region, decomp, n_n, n_idx, n_xyz):
    geo_field = iron.Field()
    # + 1
    geo_field.CreateStart(geo_field_n, region) 
    geo_field.TypeSet(iron.FieldTypes.GEOMETRIC)
    geo_field.MeshDecompositionSet(decomp)
    geo_field.VariableLabelSet(iron.FieldVariableTypes.U, "Geometry")
    # - 1
    geo_field.CreateFinish()
    # + 2
    geo_field.ParameterSetUpdateStart(
        iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES
    )
    for idx in range(n_n):
        n_id = n_idx[idx]
        n_x, n_y, n_z = (n_xyz[idx][0], n_xyz[idx][1], n_xyz[idx][2])
        # iron.Field.ParameterSetUpdateNodeDP(
        #     iron.FieldVariableTypes.U,            VARIABLE (here, displacement)
        #     iron.FieldParameterSetTypes.VALUES,   TYPE OF FIELD FOR UPDATE
        #     1,                                    VERSION OF DERIVATIVE
        #     1,                                    DERIVATIVE NUMBER
        #     n_id,                                 NODE IDENTIFIER
        #     1,                                    COMPONENT OF NODE (here, X)
        #     n_x                                   VALUE TO UPDATE TO
        # )
        geo_field.ParameterSetUpdateNodeDP(
            iron.FieldVariableTypes.U, 
            iron.FieldParameterSetTypes.VALUES,
            1, 1, n_id, X, n_x
        )
        geo_field.ParameterSetUpdateNodeDP(
            iron.FieldVariableTypes.U, 
            iron.FieldParameterSetTypes.VALUES,
            1, 1, n_id, Y, n_y
        )
        geo_field.ParameterSetUpdateNodeDP(
            iron.FieldVariableTypes.U, 
            iron.FieldParameterSetTypes.VALUES,
            1, 1, n_id, Z, n_z
        )
    # - 2
    geo_field.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
    print('+==+ ^\_/^ GEOMETRIC FIELD COMPLETE')
    return geo_field

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Material Field:
#   The next field is the material field. This field allows
#       us to dictate the constitutive properties of the 
#       material being tested. Here we intend to use a Mooney-Rivlin
#       material so this requires two constitutive components c1 and c2.
#       Here we tell the componenets to be ELEMENT_BASED which indicates
#       that the parameters can be different between elements but not
#       between nodes. 
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def material_setup(mat_field_n, decomp, geo_field, region, c):
    mat_field = iron.Field()
    # + 1
    mat_field.CreateStart(mat_field_n, region)
    mat_field.TypeSet(iron.FieldTypes.MATERIAL)
    mat_field.MeshDecompositionSet(decomp)
    mat_field.GeometricFieldSet(geo_field)
    mat_field.VariableLabelSet(iron.FieldVariableTypes.U, "Material")
    mat_field.NumberOfComponentsSet(iron.FieldVariableTypes.U, 2)
    mat_field.ComponentInterpolationSet(
        iron.FieldVariableTypes.U, 1,
        iron.FieldInterpolationTypes.ELEMENT_BASED
    )
    mat_field.ComponentInterpolationSet(
        iron.FieldVariableTypes.U, 2,
        iron.FieldInterpolationTypes.ELEMENT_BASED
    )
    # - 1
    mat_field.CreateFinish()
    mat_field.ComponentValuesInitialiseDP(
        iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, 
        1, c[0]
    )
    mat_field.ComponentValuesInitialiseDP(
        iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, 
        2, c[1]
    )
    print('+==+ ^\_/^ MATERIAL FIELD COMPLETE')
    return mat_field

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Dependent Field:
#   To hold the values of our nodes after deformation we require a 
#       dependent field. To intialise this before solving we first
#       set the values to be that of the undeformed geometry from
#       the geometric field. And add a fourth component which is 
#       indicating the hydrostatic pressure. Here, we denote 
#       ELEMENT_BASED for pressure, meaning it is per element. 
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def dependent_setup(dep_field_n, region, decomp, geo_field):
    dep_field = iron.Field()
    # + 1
    dep_field.CreateStart(dep_field_n, region)
    dep_field.MeshDecompositionSet(decomp)
    dep_field.TypeSet(iron.FieldTypes.GEOMETRIC_GENERAL)
    dep_field.GeometricFieldSet(geo_field)
    dep_field.DependentTypeSet(iron.FieldDependentTypes.DEPENDENT)
    dep_field.VariableLabelSet(iron.FieldVariableTypes.U, "Dependent")
    dep_field.NumberOfVariablesSet(2)
    dep_field.NumberOfComponentsSet(iron.FieldVariableTypes.U, 4)
    dep_field.NumberOfComponentsSet(iron.FieldVariableTypes.DELUDELN, 4)  
    dep_field.ComponentInterpolationSet(
        iron.FieldVariableTypes.U, 4,
        iron.FieldInterpolationTypes.ELEMENT_BASED
    )
    dep_field.ComponentInterpolationSet(
        iron.FieldVariableTypes.DELUDELN, 4,
        iron.FieldInterpolationTypes.ELEMENT_BASED
    )
    # - 1
    dep_field.CreateFinish()
    iron.Field.ParametersToFieldParametersComponentCopy(
        geo_field, iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, X,
        dep_field, iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, X
    )
    iron.Field.ParametersToFieldParametersComponentCopy(
        geo_field, iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, Y,
        dep_field, iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, Y
    )
    iron.Field.ParametersToFieldParametersComponentCopy(
        geo_field, iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, Z,
        dep_field, iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, Z
    )
    iron.Field.ComponentValuesInitialiseDP(
        dep_field, iron.FieldVariableTypes.U, 
        iron.FieldParameterSetTypes.VALUES, P, 0.0
    )
    print('+==+ ^\_/^ DEPENDENT FIELD COMPLETE')
    return dep_field

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Equations Field:
#    This field allows us to indicate what type of equations are used 
#       to solve the problem set. ELASTICITY indicates that the geometry
#       is being tested for an elastic case; FINITE_ELASTICITY is the 
#       the type of the FEM equation being solved. We then classify the 
#       constitutive equation for the material properties defined
#       earlier, MOONEY_RIVLIN. We can then explicitly link the other
#       fields and define output and sparsity types for solving.
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def equations_setup(eqs_field_n, eqs_set_n, region, geo_field, dep_field_n, dep_field, mat_field_n, mat_field):
    eqs_set_field = iron.Field()
    eqs_set = iron.EquationsSet()
    eqs_set_specs = [
        iron.ProblemClasses.ELASTICITY,
        iron.ProblemTypes.FINITE_ELASTICITY,
        iron.EquationsSetSubtypes.MOONEY_RIVLIN
    ]
    # + 1
    eqs_set.CreateStart(
        eqs_set_n, region, geo_field, 
        eqs_set_specs, eqs_field_n, eqs_set_field
    )
    # + 2
    eqs_set.DependentCreateStart(dep_field_n, dep_field)
    # - 2
    eqs_set.DependentCreateFinish()
    # + 3
    eqs_set.MaterialsCreateStart(mat_field_n, mat_field)
    # - 3
    eqs_set.MaterialsCreateFinish()
    # - 1
    eqs_set.CreateFinish()
    eqs = iron.Equations()
    # + 4
    eqs_set.EquationsCreateStart(eqs)
    eqs.SparsityTypeSet(iron.EquationsSparsityTypes.SPARSE)
    eqs.OutputTypeSet(iron.EquationsOutputTypes.NONE)
    # - 4
    eqs_set.EquationsCreateFinish()
    print('+==+ ^\_/^ EQUATIONS FIELD COMPLETE')
    return eqs_set_field, eqs_set, eqs

# +==+ ^\_/^ +==+ ^\_/^ +==+ 
# Problem Solver:
#    
# +==+ ^\_/^ +==+ ^\_/^ +==+ 

def problem_solver_setup(problem_n, eqs_set, loadsteps):
    problem = iron.Problem()
    problems_specs = (
        [
            iron.ProblemClasses.ELASTICITY,
            iron.ProblemTypes.FINITE_ELASTICITY,
            iron.ProblemSubtypes.NONE
        ]
    )
    # + 1
    problem.CreateStart(problem_n, problems_specs)
    # - 1
    problem.CreateFinish()
    # + 2
    problem.ControlLoopCreateStart()
    ctrl_loop = iron.ControlLoop()
    problem.ControlLoopGet(
        [iron.ControlLoopIdentifiers.NODE], 
        ctrl_loop
    )
    ctrl_loop.MaximumIterationsSet(loadsteps)
    # - 2
    problem.ControlLoopCreateFinish()
    print('+==+ ^\_/^ PROBLEM SETUP COMPLETE')
    non_solver = iron.Solver()
    lin_solver = iron.Solver()
    # + 3
    problem.SolversCreateStart()
    problem.SolverGet(
        [iron.ControlLoopIdentifiers.NODE], 
        1, 
        non_solver
    )
    non_solver.OutputTypeSet(iron.SolverOutputTypes.PROGRESS)
    non_solver.NewtonJacobianCalculationTypeSet(
        iron.JacobianCalculationTypes.EQUATIONS
    )
    non_solver.NewtonLinearSolverGet(lin_solver)
    lin_solver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
    # - 3
    problem.SolversCreateFinish()
    # 
    solver = iron.Solver()
    solver_eqs = iron.SolverEquations()
    # + 4
    problem.SolverEquationsCreateStart()
    problem.SolverGet(
        [iron.ControlLoopIdentifiers.NODE], 
        1, solver
        )
    solver.SolverEquationsGet(solver_eqs)
    solver_eqs.SparsityTypeSet(iron.SolverEquationsSparsityTypes.SPARSE)
    _ = solver_eqs.EquationsSetAdd(eqs_set)
    # - 4
    problem.SolverEquationsCreateFinish()
    print('+==+ ^\_/^ SOLVER SETUP COMPLETE')
    return problem, solver, solver_eqs

def boundary_conditions_setup(solver_eqs, dep_field, n_n, n_np_xyz):
    bcs = iron.BoundaryConditions()
    solver_eqs.BoundaryConditionsCreateStart(bcs)
    for i in range(0, n_n, 1):
        if (n_np_xyz[i, 2] == np.min(n_np_xyz[:, 2])) or (n_np_xyz[i, 2] == np.max(n_np_xyz[:, 2])):
            bcs.AddNode(
                dep_field, 
                iron.FieldVariableTypes.U,
                1,
                1,
                i+1,
                Z,
                iron.BoundaryConditionsTypes.FIXED,
                0.0
            )

        # vecNorm = np.linalg.norm([n_np_xyz[i, 0], n_np_xyz[i, 1]])
        # if abs(vecNorm - INNER_RAD) < 1e-5:
        #     bcs.AddNode(
        #         dep_field, 
        #         iron.FieldVariableTypes.U,
        #         1,
        #         1,
        #         i,
        #         X,
        #         iron.BoundaryConditionsTypes.PRESSURE,
        #         1.5
        #     )
        #     bcs.AddNode(
        #         dep_field, 
        #         iron.FieldVariableTypes.U,
        #         1,
        #         1,
        #         i,
        #         Y,
        #         iron.BoundaryConditionsTypes.PRESSURE,
        #         1.5
        #     )
        #     bcs.AddNode(
        #         dep_field, 
        #         iron.FieldVariableTypes.U,
        #         1,
        #         1,
        #         i,
        #         Z,
        #         iron.BoundaryConditionsTypes.PRESSURE,
        #         1.5
        #     )
    print('+==+ ^\_/^ BOUNDARY CONDITIONS COMPLETE')
    solver_eqs.BoundaryConditionsCreateFinish()
    return bcs, solver_eqs

def output(region, decomp, dep_field, mesh, geo_field, n_n, e_n, mesh_e):

    def_field = iron.Field()
    def_field.CreateStart(def_field_n, region)
    def_field.MeshDecompositionSet(decomp)
    def_field.TypeSet(iron.FieldTypes.GEOMETRIC)
    def_field.VariableLabelSet(iron.FieldVariableTypes.U, "DeformedGeometry")
    for component in [1, 2, 3]:
        def_field.ComponentMeshComponentSet(
                iron.FieldVariableTypes.U, component, 1)
    def_field.CreateFinish()

    pre_field = iron.Field()
    pre_field.CreateStart(pre_field_n, region)
    pre_field.MeshDecompositionSet(decomp)
    pre_field.VariableLabelSet(iron.FieldVariableTypes.U, "Pressure")
    pre_field.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 1, 1)
    pre_field.ComponentInterpolationSet(
        iron.FieldVariableTypes.U, 
        1, 
        iron.FieldInterpolationTypes.ELEMENT_BASED
    )
    pre_field.NumberOfComponentsSet(iron.FieldVariableTypes.U, 1)
    pre_field.CreateFinish()

    for component in [1, 2, 3]:
        dep_field.ParametersToFieldParametersComponentCopy(
            iron.FieldVariableTypes.U,
            iron.FieldParameterSetTypes.VALUES, component,
            def_field, iron.FieldVariableTypes.U,
            iron.FieldParameterSetTypes.VALUES, component)

    dep_field.ParametersToFieldParametersComponentCopy(
        iron.FieldVariableTypes.U,
        iron.FieldParameterSetTypes.VALUES,
        4,
        pre_field, 
        iron.FieldVariableTypes.U,
        iron.FieldParameterSetTypes.VALUES, 
        1
    )

    meshNodes = iron.MeshNodes()
    mesh.NodesGet(1,meshNodes)

    n_list = []
    n_bef = []
    n_aft = []

    for i in range(0, n_n, 1):
        n_bef.append(
            [
                geo_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1,
                    1,
                    i+1,
                    1
                ),
                geo_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1,
                    1,
                    i+1,
                    2
                ),
                geo_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1,
                    1,
                    i+1,
                    3
                )
            ]
        )
        n_aft.append(
            [
                dep_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1,
                    1,
                    i+1,
                    1
                ),
                dep_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1,
                    1,
                    i+1,
                    2
                ),
                dep_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1,
                    1,
                    i+1,
                    3
                )
            ]
        )
        n_list.append(
            [
                i+1, 
                n_bef[i][0], 
                n_bef[i][1],
                n_bef[i][2], 
                n_aft[i][0], 
                n_aft[i][1], 
                n_aft[i][2]
            ]
        )

    n_file = open('/home/jovyan/work/test_files/runtime_files/output_mesh.node', 'w')
    n_file.writelines([str(line) + "\n" for line in n_list])
    n_file.close()

    e_list = []
    for j in range(0, e_n, 1):
        elem_n = mesh_e.NodesGet(1+j, 27)
        e_list[j].append(elem_n)

    elem_file=open('/home/jovyan/work/test_files/runtime_files/output_mesh.ele','w')
    elem_file.writelines([str(line) + "\n" for line in e_list])
    elem_file.close()

    before_def = np.array(n_bef)
    np.save('/home/jovyan/work/test_files/runtime_files/output_mesh_before_def.npy',before_def)
    after_def = np.array(n_aft)
    np.save('/home/jovyan/work/test_files/runtime_files/output_mesh_after_def.npy',after_def)

    points = before_def
    point_data = {"deformed": after_def}

    cells_array = np.array(e_list)[:,:] - 1
    cells = [("tetra10", cells_array)] + [("tetra10", cells_array)]

    meshio.write_points_cells("/home/jovyan/work/test_files/runtime_files/output_mesh.vtk",points, cells, point_data)

    meshio.write_points_cells("/home/jovyan/work/test_files/runtime_files/output_mesh_iron.vtk",points,cells, point_data)
    print('+==+ ^\_/^ EXPORT COMPLETE')
    return