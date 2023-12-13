#!/usr/bin/env python

#> \file
#> \author Vijay Rajagopal, Liam Murray
#> \brief This is an example script to solve a finite elasticity equation using OpenCMISS calls in python.
#>
#> \section LICENSE
#>
#> Version: MPL 1.1/GPL 2.0/LGPL 2.1
#>
#> The contents of this file are subject to the Mozilla Public License
#> Version 1.1 (the "License"); you may not use this file except in
#> compliance with the License. You may obtain a copy of the License at
#> http://www.mozilla.org/MPL/
#>
#> Software distributed under the License is distributed on an "AS IS"
#> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#> License for the specific language governing rights and limitations
#> under the License.
#>
#> The Original Code is OpenCMISS
#>
#> The Initial Developer of the Original Code is University of Auckland,
#> Auckland, New Zealand and University of Oxford, Oxford, United
#> Kingdom. Portions created by the University of Auckland and University
#> of Oxford are Copyright (C) 2007 by the University of Auckland and
#> the University of Oxford. All Rights Reserved.
#>
#> Contributor(s): 
#>
#> Alternatively, the contents of this file may be used under the terms of
#> either the GNU General Public License Version 2 or later (the "GPL"), or
#> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
#> in which case the provisions of the GPL or the LGPL are applicable instead
#> of those above. if you wish to allow use of your version of this file only
#> under the terms of either the GPL or the LGPL, and not to allow others to
#> use your version of this file under the terms of the MPL, indicate your
#> decision by deleting the provisions above and replace them with the notice
#> and other provisions required by the GPL or the LGPL. if you do not delete
#> the provisions above, a recipient may use your version of this file under
#> the terms of any one of the MPL, the GPL or the LGPL.
#>

#> \example FiniteElasticity/UniAxialExtension/src/UniAxialExtensionExample.py
## Example script to solve a finite elasticity equation using OpenCMISS calls in python.
## \par Latest Builds:
## \li <a href='http://autotest.bioeng.auckland.ac.nz/opencmiss-build/logs_x86_64-linux/FiniteElasticity/UniAxialExtension/build-intel'>Linux Intel Build</a>
## \li <a href='http://autotest.bioeng.auckland.ac.nz/opencmiss-build/logs_x86_64-linux/FiniteElasticity/UniAxialExtension/build-gnu'>Linux GNU Build</a>
#<

#> Main script
# Add Python bindings directory to PATH

#sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

# Intialise OpenCMISS
from opencmiss.iron import iron
import math
import numpy as np
import meshio
import generateMesh
import scipy.optimize
import matplotlib.pyplot as plt

# VIJAY_IRON = [
#     6, 3, 0, 15, 12, 9, 24, 21, 18,
#     7, 4, 1, 16, 13, 10, 25, 22, 19,
#     8, 5, 2, 17, 14, 11, 26, 23, 20
# ]

# VIJAY_IRON = [
#     0, 9, 18, 3, 12, 21, 6, 15, 24,
#     1, 10, 19, 4, 13, 22, 7, 16, 25,
#     2, 11, 20, 5, 14, 23, 8, 17, 26
# ]

# VIJAY_IRON = [
#     0, 3, 6, 9, 12, 15, 18, 21, 24,
#     1, 4, 7, 10, 13, 16, 19, 22, 25,
#     2, 5, 8, 11, 14, 17, 20, 23, 26
# ]

# VIJAY_IRON = [
#     6, 15, 24, 3, 12, 21, 0, 9, 18,
#     7, 16, 25, 4, 13, 22, 1, 10, 19,
#     8, 17, 26, 5, 14, 23, 2, 11, 20
# ]

#Setting up the vtk output function before use after solve.

def vtkoutput(filenameroot,totalNumberOfNodes, totalNumberOfElements, mesh,geo_field,dep_field):
    X = 1
    Y = 2
    Z = 3
    meshNodes = iron.MeshNodes()
    mesh.NodesGet(1,meshNodes)
    n_list = []
    n_bef = []
    n_aft = []
    for i in range(0, totalNumberOfNodes, 1):
        n_bef.append(
            [
                geo_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, 1, i+1, X
                ),
                geo_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, 1, i+1, Y
                ),
                geo_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, 1, i+1, Z
                )
            ]
        )


        n_aft.append(
            [
                dep_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, 1, i+1, X
                ),
                dep_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, 1, i+1, Y
                ),
                dep_field.ParameterSetGetNodeDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES,
                    1, 1, i+1, Z
                )
            ]
        )
        n_list.append(
            [
                i+1,
                n_bef[i][0], n_bef[i][1], n_bef[i][2],
                n_aft[i][0], n_aft[i][1], n_aft[i][2]
            ]
        )
    # +============+
    # Store elements
    # +============+
    meshElements = iron.MeshElements()
    mesh.ElementsGet(1,meshElements)

    e_list = []
    for i in range(0, totalNumberOfElements, 1):
        e_list.append(meshElements.NodesGet(i+1, 27))

    e_list_iron = np.array(e_list)[:,:]-1

    print(e_list_iron)


    # Iron Numbering for Hexa-27
    #   *  z = 0           z = 0.5         z = 1          
    #   *  6--7 --8     * 15--16--17    x 24--25--26
    #   *  |      |     *  |      |     x  |      |
    #   *  3  4   5     * 12  13  14    x 21  22  23
    #   *  |      |     *  |      |     x  |      |
    #   *  0--1 --2     *  9--10--11    x 18--19--20

    # Vijay Numbering for Hexa-27
    #   *  z = 0           z = 0.5         z = 1    
    #   *  0-- 9--18    *  1--10--19    *  2--11--20     
    #   *  |      |     *  |      |     *  |      |       
    #   *  3  12  21    *  4  13  22    *  5  14  23     
    #   *  |      |     *  |      |     *  |      |     
    #   *  6--15--24    *  7--16--25    *  8--17--26      

    # VTK Numbering for Hexa-27
    #   *  z = 0           z = 0.5         z = 1    
    #   *  3--10--2     * 19--23--18    x  7--14--6
    #   *  |      |     *  |      |     x  |      |
    #   * 11  24  9     * 20  26  21    x 15  25  13
    #   *  |      |     *  |      |     x  |      |
    #   *  0-- 8--1     * 16--22--17    x  4--12--5 

    VIJAY_VTK = [
        6, 24, 18, 0, 8, 26, 20, 2, 
        15, 21, 9, 3, 17, 23, 11, 5, 
        7, 25, 19, 1, 4, 22, 16, 10, 
        12, 14, 13
    ]


    e_list_vtk = e_list_iron[:, VIJAY_VTK]

    # +============+
    # Store data files
    # +============+
    filename = filenameroot+'.node'
    node_file = open(filename, 'w')
    node_file.writelines([str(line) + "\n" for line in n_list])
    node_file.close()
    filename = filenameroot+'.ele'
    elem_file=open(filename,'w')
    elem_file.writelines([str(line) + "\n" for line in e_list])
    elem_file.close()
    bef_def = np.array(n_bef)
    aft_def = np.array(n_aft)
    filename = filenameroot+'_mesh_before_def.npy'
    np.save(filename,bef_def)
    filename = filenameroot+'_mesh_after_def.npy'
    np.save(filename,aft_def)
    # +============+
    # VTK export
    # +============+
    filename = filenameroot+'.vtk'
    print([("hexahedron27", e_list_vtk)] + [("hexahedron27", e_list_vtk)])
    meshio.write_points_cells(
        filename,
        bef_def,
        [("hexahedron27", e_list_vtk)], #+ [("hexahedron27", e_list_vtk)],
        {"deformed": aft_def-bef_def}
    )

    return

# Set problem parameters
#cylinder geometry inputs
r_inner = 1.0
r_outer = 1.5
z_length = 1.0
#material parameters
c10 = 2.0
c01 = 6.0
#loading conditions
pressure_internal = 1.5
pressure_external = 0.0
stretch_ratio = 1.2
twist_angle = 0.0 #math.pi/6.0  #in radians
#mesh parameters
#set number of elements in each direction of cylinder by setting refined_mesh_option.
#set refinement option stored in a list as [radial,circumferential,z]
refined_mesh = [[1,4,1],[1,8,1],[2,4,1],[2,8,1],[2,8,2],[4,16,2],[4,16,8],[4,32,16],[8,64,16]]
refined_mesh_option = 7
numberOfRadialElements = refined_mesh[refined_mesh_option][0]
numberOfCircumferentialElements = refined_mesh[refined_mesh_option][1]
numberOfZElements = refined_mesh[refined_mesh_option][2]
#set geometric basis interpolation to one of:
# 1 - linear lagrange
# 2 - quadratic lagrange
# 3 - cubic lagrange
# 4 - hermite cubic
# 5 - hermite quintic
# 6 - hermite septum
# 7 - linear simplex
# 8 - quadratic simplex
# 9 - cubic simplex

InterpolationType = 2
#decide if you want to use a pressure basis for the pressure field
UsePressureBasis = False
numberOfLoadIncrements = 3
#From here the code sets up the problem with the inputs from above.

# Set OpenCMISS parameters
coordinateSystemUserNumber = 1
regionUserNumber = 1
geometricbasisUserNumber = 1

pressureBasisUserNumber = 2
generatedMeshUserNumber = 1
meshUserNumber = 1
decompositionUserNumber = 1
geometricFieldUserNumber = 1
fibreFieldUserNumber = 2
materialFieldUserNumber = 3
dependentFieldUserNumber = 4
equationsSetUserNumber = 1
equationsSetFieldUserNumber = 5
problemUserNumber = 1
dataPointsUserNumber=1
dataProjectionUserNumber=1
# Get the number of computational nodes and this computational node number
numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
computationalNodeNumber = iron.ComputationalNodeNumberGet()

# Create a 3D rectangular cartesian coordinate system
coordinateSystem = iron.CoordinateSystem()
coordinateSystem.CreateStart(coordinateSystemUserNumber)
coordinateSystem.DimensionSet(3)
coordinateSystem.CreateFinish()

# Create a region and assign the coordinate system to the region
region = iron.Region()
region.CreateStart(regionUserNumber,iron.WorldRegion)
region.LabelSet("Region")
region.coordinateSystem = coordinateSystem
region.CreateFinish()

# Define bases

#setting up basis function related settings and accordingly number of nodes and number of elements.
if(InterpolationType==2):
    totalNumberOfElements = numberOfRadialElements*numberOfCircumferentialElements*numberOfZElements
    if InterpolationType==1:
        totalNumberOfNodes = 8*totalNumberOfElements-(4*totalNumberOfElements)
    elif InterpolationType==2:
        totalNumberOfNodes = 27*totalNumberOfElements-(9*totalNumberOfElements)
if(UsePressureBasis):
    numberOfMeshComponents = 2
    if InterpolationType==1:
        totalNumberOfPressureNodes = 8*totalNumberOfElements-(4*totalNumberOfElements)
    elif (InterpolationType==2):
        print("Using quadratic lagrange pressure basis is not supported, please change to linear interpolation or no pressure basis (only element varying)")
        exit()
else:
    numberOfMeshComponents = 1

#hard coding two parameters because the cylinder is 3D and lower number of gauss points per xi direction than 4 leads to significant model simulation errors.
numberOfXi = 3
NumberOfGaussXi = 4

geometricBasis = iron.Basis()
geometricBasis.CreateStart(geometricbasisUserNumber)
if InterpolationType in (1,2,3,4):
    geometricBasis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
elif InterpolationType in (7,8,9):
    geometricBasis.type = iron.BasisTypes.SIMPLEX
geometricBasis.numberOfXi = numberOfXi
if (InterpolationType==1):
    geometricBasis.interpolationXi = [iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfXi
elif (InterpolationType==2):
    geometricBasis.interpolationXi = [iron.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfXi

geometricBasis.quadratureNumberOfGaussXi = [NumberOfGaussXi]*numberOfXi
geometricBasis.CreateFinish()

if(UsePressureBasis):
    # Define pressure basis
    pressureBasis = iron.Basis()
    pressureBasis.CreateStart(pressureBasisUserNumber)
    if InterpolationType in (1,2,3,4):
        pressureBasis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
    elif InterpolationType in (7,8,9):
        pressureBasis.type = iron.BasisTypes.SIMPLEX
    pressureBasis.numberOfXi = numberOfXi
    if InterpolationType==1:
        pressureBasis.interpolationXi = [iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfXi
    elif InterpolationType==2: #this should be unnecessary because i have caught this case above
        print("Using quadratic lagrange pressure basis is not supported, please change to linear interpolation or no pressure basis (only element varying)")
        exit()
    if(NumberOfGaussXi>0):
        pressureBasis.quadratureNumberOfGaussXi = [NumberOfGaussXi]*numberOfXi
    pressureBasis.CreateFinish()

# Start the creation of a manually generated mesh in the region
mesh = iron.Mesh()
mesh.CreateStart(meshUserNumber,region,numberOfXi)
mesh.NumberOfComponentsSet(numberOfMeshComponents)
mesh.NumberOfElementsSet(totalNumberOfElements)

#Define nodes for the mesh
nodes = iron.Nodes()
nodes.CreateStart(region,totalNumberOfNodes)
nodes.CreateFinish()
(
    node_list, node_idx_list, top_node_list, 
    bottom_node_list, yfix_node_list, xfix_node_list, 
    internal_node_list, outer_node_list, e_assign
) = generateMesh.annulus(r_inner, r_outer, z_length, numberOfRadialElements, numberOfCircumferentialElements, numberOfZElements, InterpolationType)


# import matplotlib.pyplot as plt
# # Create the 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i, k in enumerate(e_assign[3, :]):
#     print(node_list[int(k)-1])
#     x = node_list[int(k)-1][0]
#     y = node_list[int(k)-1][1]
#     z = node_list[int(k)-1][2]
#     ax.text(x, y, z, str(int(k)), color='red')
#     ax.scatter(x, y, z, s=50)

# # Set labels for the axes
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')

# for ii in range(0,360,90):
#         ax.view_init(elev=10., azim=ii)
#         plt.savefig("movie%d.png" % ii)

# print(tea)
# Define elements for the mesh
elements = iron.MeshElements()
meshComponentNumber = 1
elements.CreateStart(mesh,meshComponentNumber,geometricBasis)

for elemidx in range(0, len(e_assign[:,0]), 1):
    elemNum = elemidx+1
    # elemNodes = np.array(e_assign[elemidx, VIJAY_IRON], dtype=np.int32)
    print('element number:', int(elemNum))
    elemNodes = np.array(e_assign[elemidx], dtype=np.int32)
    # print('listing each element nodes')
    elements.NodesSet(int(elemNum), elemNodes)

elements.CreateFinish()
mesh.CreateFinish() 

# Create a decomposition for the mesh
decomposition = iron.Decomposition()
decomposition.CreateStart(decompositionUserNumber,mesh)
decomposition.type = iron.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
decomposition.CreateFinish()

# Create a field for the geometry
geometricField = iron.Field()
geometricField.CreateStart(geometricFieldUserNumber,region)
geometricField.MeshDecompositionSet(decomposition)
geometricField.TypeSet(iron.FieldTypes.GEOMETRIC)
geometricField.VariableLabelSet(iron.FieldVariableTypes.U,"Geometry")
geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,3,1)
if InterpolationType == 4:
    geometricField.fieldScalingType = iron.FieldScalingTypes.ARITHMETIC_MEAN
geometricField.CreateFinish()

# Update the geometric field parameters manually for a cube with a cube-shaped hole in it.
#outer cube unit length = 2.0 mm
#inner cubic hole unit length = 1.0 mm
#Number of nodes = 16
geometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
print('Total nodes in mesh=',len(node_list))
for nodeidx in range(0,len(node_list),1):
    # print(nodeidx)
    nodeNum = nodeidx+1
    nodex = node_list[nodeidx][0]
    nodey = node_list[nodeidx][1]
    nodez = node_list[nodeidx][2]
    geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,nodeNum,1,nodex)
    geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,nodeNum,2,nodey)
    geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,nodeNum,3,nodez)

geometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)

# Create a fibre field and attach it to the geometric field
fibreField = iron.Field()
fibreField.CreateStart(fibreFieldUserNumber,region)
fibreField.TypeSet(iron.FieldTypes.FIBRE)
fibreField.MeshDecompositionSet(decomposition)
fibreField.GeometricFieldSet(geometricField)
fibreField.VariableLabelSet(iron.FieldVariableTypes.U,"Fibre")
if InterpolationType == 4:
    fibreField.fieldScalingType = iron.FieldScalingTypes.ARITHMETIC_MEAN
fibreField.CreateFinish()

# Create the material field
materialField = iron.Field()
materialField.CreateStart(materialFieldUserNumber,region)
materialField.TypeSet(iron.FieldTypes.MATERIAL)
materialField.MeshDecompositionSet(decomposition)
materialField.GeometricFieldSet(geometricField)
materialField.NumberOfVariablesSet(1)
materialField.NumberOfComponentsSet(iron.FieldVariableTypes.U,2)
materialField.VariableLabelSet(iron.FieldVariableTypes.U,"Material")
materialField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
materialField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
if InterpolationType == 4:
    materialField.fieldScalingType = iron.FieldScalingTypes.ARITHMETIC_MEAN
materialField.CreateFinish()

# Set Mooney-Rivlin constants c10 and c01 respectively.
iron.Field.ComponentValuesInitialiseDP(
    materialField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,c01)
iron.Field.ComponentValuesInitialiseDP(
    materialField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,c10)

# Create the dependent field
dependentField = iron.Field()
dependentField.CreateStart(dependentFieldUserNumber,region)
dependentField.VariableLabelSet(iron.FieldVariableTypes.U,"Dependent")
dependentField.TypeSet(iron.FieldTypes.GEOMETRIC_GENERAL)  
dependentField.MeshDecompositionSet(decomposition)
dependentField.GeometricFieldSet(geometricField) 
dependentField.DependentTypeSet(iron.FieldDependentTypes.DEPENDENT) 
dependentField.NumberOfVariablesSet(2)
dependentField.NumberOfComponentsSet(iron.FieldVariableTypes.U,4)
dependentField.NumberOfComponentsSet(iron.FieldVariableTypes.DELUDELN,4)
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,3,1)  
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,1,1)
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,2,1)
dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,3,1)  
dependentField.ComponentInterpolationSet(iron.FieldVariableTypes.U,4,iron.FieldInterpolationTypes.ELEMENT_BASED)
dependentField.ComponentInterpolationSet(iron.FieldVariableTypes.DELUDELN,4,iron.FieldInterpolationTypes.ELEMENT_BASED)
if(UsePressureBasis):
    # Set the pressure to be nodally based and use the second mesh component
    if InterpolationType == 4:
        dependentField.ComponentInterpolationSet(iron.FieldVariableTypes.U,4,iron.FieldInterpolationTypes.NODE_BASED)
        dependentField.ComponentInterpolationSet(iron.FieldVariableTypes.DELUDELN,4,iron.FieldInterpolationTypes.NODE_BASED)
    dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,4,2)
    dependentField.ComponentMeshComponentSet(iron.FieldVariableTypes.DELUDELN,4,2)
if InterpolationType == 4:
    dependentField.fieldScalingType = iron.FieldScalingTypes.ARITHMETIC_MEAN
dependentField.CreateFinish()

# Initialise dependent field from undeformed geometry and displacement bcs and set hydrostatic pressure
iron.Field.ParametersToFieldParametersComponentCopy(
    geometricField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,
    dependentField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1)
iron.Field.ParametersToFieldParametersComponentCopy(
    geometricField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,
    dependentField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2)
iron.Field.ParametersToFieldParametersComponentCopy(
    geometricField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,3,
    dependentField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,3)
iron.Field.ComponentValuesInitialiseDP(
    dependentField,iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,4,1.2345678806304932)

# Create the equations_set
equationsSetField = iron.Field()
equationsSet = iron.EquationsSet()
equationsSetSpecification = [iron.EquationsSetClasses.ELASTICITY,
    iron.EquationsSetTypes.FINITE_ELASTICITY,
    iron.EquationsSetSubtypes.MOONEY_RIVLIN]
equationsSet.CreateStart(equationsSetUserNumber,region,fibreField,
    equationsSetSpecification, equationsSetFieldUserNumber, equationsSetField)
equationsSet.CreateFinish()

equationsSet.MaterialsCreateStart(materialFieldUserNumber,materialField)
equationsSet.MaterialsCreateFinish()

equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
equationsSet.DependentCreateFinish()

# Create equations
equations = iron.Equations()
equationsSet.EquationsCreateStart(equations)
equations.sparsityType = iron.EquationsSparsityTypes.SPARSE
equations.outputType = iron.EquationsOutputTypes.NONE
equationsSet.EquationsCreateFinish()

# Define the problem
problem = iron.Problem()
problemSpecification = [iron.ProblemClasses.ELASTICITY,
        iron.ProblemTypes.FINITE_ELASTICITY,
        iron.ProblemSubtypes.NONE]
problem.CreateStart(problemUserNumber, problemSpecification)
problem.CreateFinish()

# Create control loops
problem.ControlLoopCreateStart()
controlLoop = iron.ControlLoop()
problem.ControlLoopGet([iron.ControlLoopIdentifiers.NODE],controlLoop)
controlLoop.MaximumIterationsSet(numberOfLoadIncrements)
problem.ControlLoopCreateFinish()

# Create problem solver
nonLinearSolver = iron.Solver()
linearSolver = iron.Solver()
problem.SolversCreateStart()
problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,nonLinearSolver)
nonLinearSolver.outputType = iron.SolverOutputTypes.PROGRESS
nonLinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.FD)
nonLinearSolver.NewtonLinearSolverGet(linearSolver)
linearSolver.linearType = iron.LinearSolverTypes.DIRECT
#linearSolver.libraryType = iron.SolverLibraries.LAPACK
problem.SolversCreateFinish()

# Create solver equations and add equations set to solver equations
solver = iron.Solver()
solverEquations = iron.SolverEquations()
problem.SolverEquationsCreateStart()
problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,solver)
solver.SolverEquationsGet(solverEquations)
solverEquations.sparsityType = iron.SolverEquationsSparsityTypes.SPARSE
equationsSetIndex = solverEquations.EquationsSetAdd(equationsSet)
problem.SolverEquationsCreateFinish()

# Prescribe boundary conditions (absolute nodal parameters)
boundaryConditions = iron.BoundaryConditions()
solverEquations.BoundaryConditionsCreateStart(boundaryConditions)

locked, locked_rotation_x,locked_rotation_y, extended, pressured = [], [], [], [],[]
for i in node_idx_list:
    idx = i[0]-1
    #print(node_list[idx])
    if np.isclose(node_list[idx][2], 0):
       locked.append(idx)
       boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,1,1,idx+1,3,iron.BoundaryConditionsTypes.FIXED_INCREMENTED,0.0)
       if np.isclose(node_list[idx][0], 0):
            locked_rotation_x.append(idx)
            boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,1,1,idx+1,1,iron.BoundaryConditionsTypes.FIXED_INCREMENTED,0.0)
       elif np.isclose(node_list[idx][1], 0):
            locked_rotation_y.append(idx)
            boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,1,1,idx+1,2,iron.BoundaryConditionsTypes.FIXED_INCREMENTED,0.0)
       continue
    if np.isclose(node_list[idx][2], z_length):
       extended.append(idx)
       boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,1,1,idx+1,3,iron.BoundaryConditionsTypes.FIXED_INCREMENTED,(stretch_ratio*z_length)-z_length)
       continue
    if np.isclose(np.linalg.norm([node_list[idx][0], node_list[idx][1]]), r_inner):
       pressured.append(idx)
       boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.DELUDELN,1,1,idx+1,3,iron.BoundaryConditionsTypes.PRESSURE_INCREMENTED,pressure_internal)
       continue
    if np.isclose(np.linalg.norm([node_list[idx][0], node_list[idx][1]]), r_outer):
       pressured.append(idx)
       boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.DELUDELN,1,1,idx+1,3,iron.BoundaryConditionsTypes.PRESSURE_INCREMENTED,pressure_external)
       continue

#print("LOCKED: {}".format(locked))
#print("LOCKED_ROTATION_X: {}".format(locked_rotation_x))
#print("LOCKED_ROTATION_Y: {}".format(locked_rotation_y))
#print("PRESSURED: {}".format(pressured))
#print("EXTENDED: {}".format(extended))

# for nodeidx in range(0,len(bottom_node_list),1):
#     nodeNum = bottom_node_list[nodeidx][0]
#     # print(nodeNum)
#     boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,1,1,nodeNum,3,iron.BoundaryConditionsTypes.FIXED_INCREMENTED,0.0)
# for nodeidx in range(0,len(yfix_node_list),1):
#     nodeNum = yfix_node_list[nodeidx][0]
#     boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,1,1,nodeNum,2,iron.BoundaryConditionsTypes.FIXED_INCREMENTED,0.0)
# for nodeidx in range(0,len(xfix_node_list),1):
#     nodeNum = xfix_node_list[nodeidx][0]
#     boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,1,1,nodeNum,1,iron.BoundaryConditionsTypes.FIXED_INCREMENTED,0.0)

# #Pressure boundary conditions on the internal faces.
# for nodeidx in range(0,len(internal_node_list),1):
#     nodeNum = internal_node_list[nodeidx][0]
#     boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.DELUDELN,1,1,nodeNum,3,iron.BoundaryConditionsTypes.PRESSURE_INCREMENTED,pressure_internal)

# #Pressure boundary conditions on the external faces.
# for nodeidx in range(0,len(outer_node_list),1):
#     nodeNum = outer_node_list[nodeidx][0]
#     boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.DELUDELN,1,1,nodeNum,3,iron.BoundaryConditionsTypes.PRESSURE_INCREMENTED,pressure_external)

#Apply stretch boundary condition on top faces.
# for nodeidx in range(0,len(top_node_list),1):
#     nodeNum = top_node_list[nodeidx][0]
#     stretch_disp = stretch_ratio*z_length-z_length
#     boundaryConditions.AddNode(dependentField,iron.FieldVariableTypes.U,1,1,nodeNum,3,iron.BoundaryConditionsTypes.FIXED_INCREMENTED,stretch_disp)

solverEquations.BoundaryConditionsCreateFinish()

# Solve the problem
problem.Solve()
#output deformed geometry
filenameroot = 'output_refined_mesh_'+str(refined_mesh_option)
vtkoutput(filenameroot,len(node_list),totalNumberOfElements,mesh,geometricField,dependentField)
###Generate results to check the stresses and strains
#Create lines of points to evaluate results at
# datapoints_radii = np.linspace(r_inner+0.1,r_outer-0.1,1)
# datalines_theta = np.linspace(0.1,2.0*math.pi,1)
# datalines_z = np.linspace(0.1,z_length-0.1,1)
# datalines = []
# numDataPoints = 0
# for datalineidx in range(0,1):
#     dataline_z = datalines_z[datalineidx]
#     dataline_theta = datalines_theta[datalineidx]
#     dataline = []
#     for pointidx in range(0,len(datapoints_radii),1):
#         datapoint_radius = datapoints_radii[pointidx]
#         datapoint_x = datapoint_radius*math.cos(dataline_theta)
#         datapoint_y = datapoint_radius*math.sin(dataline_theta)
#         datapoint_z = dataline_z
#         dataline.append([datapoint_x,datapoint_y,datapoint_z])
#         print('datapoint=',[datapoint_x,datapoint_y,datapoint_z])
#         numDataPoints = numDataPoints+1
#     datalines.append(dataline)
# dataPoints = iron.DataPoints()
# dataPoints.CreateStart(dataPointsUserNumber,region,numDataPoints)
#
# #convert list of points into opencmiss datapoints object
# datapointidx = 0
# for datalineidx in range(0,len(datalines),1):
#     dataline = datalines[datalineidx]
#     for pointidx in range(0,len(dataline),1):
#         datapointidx=datapointidx+1
#         datapoint = dataline[pointidx]
#         # Set up data projection
#         dataPoints.PositionSet(datapointidx,datapoint)
# dataPoints.CreateFinish()
# print('checking datapoints have been set correctly')
# for datapointidx in range(1,numDataPoints+1,1):
#     print(dataPoints.PositionGet(datapointidx,3))
# print('Project datapoints into finite element mesh using dataprojections object.')
# dataProjections=iron.DataProjection()
# dataProjections.CreateStart(dataProjectionUserNumber,dataPoints,geometricField,iron.FieldVariableTypes.U)
# dataProjections.projectionType = iron.DataProjectionProjectionTypes.ALL_ELEMENTS
# dataProjections.CreateFinish()
#
# print('evaluate data projects based on gemetric field')
# dataProjections.DataPointsProjectionEvaluate(iron.FieldParameterSetTypes.VALUES)
# # Create mesh topology for data projection
# mesh.TopologyDataPointsCalculateProjection(dataProjections)
# # Create decomposition topology for data projection
# decomposition.TopologyDataProjectionCalculate()
# dataProjections.ResultAnalysisOutput("")
#
# print('for each dataline, extract the tensors and invariants and store them in a dictionary')
# datalines_sigma_xx, datalines_sigma_yy, datalines_sigma_zz, datalines_sigma_xy, datalines_sigma_xz, datalines_sigma_yz = [],[],[],[],[],[]
# for datalineidx in range(0,len(datalines),1):
#     dataline = datalines[datalineidx]
#     dataline_sigma_xx,dataline_sigma_yy,dataline_sigma_zz,dataline_sigma_xy,dataline_sigma_xz,dataline_sigma_yz = [],[],[],[],[],[]
#     for pointidx in range(0,len(dataline),1):
#         datapointidx = datalineidx*len(dataline)+pointidx+1
#         xiPosition = dataProjections.ResultXiGet(datapointidx, 3)
#         elementNumber = dataProjections.ResultElementNumberGet(datapointidx)
#         TC = equationsSet.TensorInterpolateXi(iron.EquationsSetDerivedTensorTypes.CAUCHY_STRESS,elementNumber, xiPosition, (3, 3))
#         dataline_sigma_xx.append(TC[0,0])
#         dataline_sigma_yy.append(TC[1,1])
#         dataline_sigma_zz.append(TC[2,2])
#         dataline_sigma_xy.append(TC[0,1])
#         dataline_sigma_xz.append(TC[0,2])
#         dataline_sigma_yz.append(TC[1,2])
#     datalines_sigma_yy.append(dataline_sigma_yy)
#     datalines_sigma_zz.append(dataline_sigma_zz)
#     datalines_sigma_xy.append(dataline_sigma_xy)
#     datalines_sigma_xz.append(dataline_sigma_xz)
#     datalines_sigma_yz.append(dataline_sigma_yz)
#     datalines_sigma_xx.append(dataline_sigma_xx)
#
# print('plot the stresses as a function of radius, saved in stress.png')
#
#
#
# fig,axes = plt.subplots(3,2)
#
# axes[0,0].plot(datapoints_radii,datalines_sigma_xx[0])
# #axes[0,0].set_ylim([-2.0,1.0])
# axes[0,0].set_title("sigma_xx")
#
# axes[0,1].plot(datapoints_radii,datalines_sigma_yy[0])
# axes[0,1].set_title("sigma_yy")
#  #axes[0,1].set_ylim([2,5])
# #
# axes[1,0].plot(datapoints_radii,datalines_sigma_zz[0])
# axes[1,0].set_title("sigma_zz")
# # #axes[1,0].set_ylim([2,8])
# #
# axes[1,1].plot(datapoints_radii,datalines_sigma_xy[0])
# axes[1,1].set_title("sigma_xy")
# # #axes[1,0].set_ylim([2,8])
#
# plt.savefig("stress.png")
#

results = {}
elementNumbers = []
if refined_mesh_option == 0:#[1,4,1]
    elementNumbers = [1]
    xipoint_radii = np.linspace(r_inner,r_outer,10)
elif refined_mesh_option == 1:# [1,8,1]
    elementNumbers = [1]
    xipoint_radii = np.linspace(r_inner,r_outer,10)
elif refined_mesh_option == 2:# [2,4,1]
    elementNumbers=[1,5]
    xipoint_radii = np.linspace(r_inner,r_outer,20)
elif refined_mesh_option == 3:# [2,8,1]
    elementNumbers=[1,9]
    xipoint_radii = np.linspace(r_inner,r_outer,20)
elif refined_mesh_option == 4:# [2,8,2]
    elementNumbers=[1,17]
    xipoint_radii = np.linspace(r_inner,r_outer,20)
elif refined_mesh_option == 5:# [4,16,2]
    elementNumbers=[1,33,65,97]
    xipoint_radii = np.linspace(r_inner,r_outer,40)
elif refined_mesh_option == 6:# [4,16,8]
    elementNumbers=[1,129,257,385]
    xipoint_radii = np.linspace(r_inner,r_outer,40)
elif refined_mesh_option == 7:# [4,32,16]
    elementNumbers=[1,513,1025,1537]
    xipoint_radii = np.linspace(r_inner,r_outer,40)
elif refined_mesh_option == 8:# [8,64,16]
    elementNumbers=[1,513,1023,1537,2048,2561,3073,3585]
    xipoint_radii = np.linspace(r_inner,r_outer,80)
xis = []
xi_3s = np.linspace(0, 1, 10)
for xipoint in range(0, len(xi_3s), 1):
    xi_3 = xi_3s[xipoint] #note that xi3 is the radial direction through the wall of each element
    xi_2 = 0.5 #note that xi2 is the circumferential direction around the cylinder
    xi_1 = 0.5 #note that xi1 is the axial direction along the cylinder
    xis.append([xi_1, xi_2, xi_3])

dataline_sigma_xx, dataline_sigma_yy, dataline_sigma_zz, dataline_sigma_xy, dataline_sigma_xz, dataline_sigma_yz = [],[],[],[],[],[]

for elementNumber in elementNumbers:
    for xipoint in range(0, len(xis), 1):
        xiPosition = xis[xipoint]
        TC = equationsSet.TensorInterpolateXi(iron.EquationsSetDerivedTensorTypes.CAUCHY_STRESS,elementNumber, xiPosition, (3, 3))
        dataline_sigma_xx.append(TC[0,0])
        dataline_sigma_yy.append(TC[1,1])
        dataline_sigma_zz.append(TC[2,2])
        dataline_sigma_xy.append(TC[0,1])
        dataline_sigma_xz.append(TC[0,2])
        dataline_sigma_yz.append(TC[1,2])


print('plot the stresses as a function of radius, saved in stress png file')
#
#
#
fig,axes = plt.subplots(3,2)
fig.suptitle("Stress vs Radius, Refined Mesh Option "+str(refined_mesh[refined_mesh_option]))

#
axes[0,0].plot(xipoint_radii,dataline_sigma_xx)
# #axes[0,0].set_ylim([-2.0,1.0])
axes[0,0].set_title("sigma_xx")
#
axes[0,1].plot(xipoint_radii,dataline_sigma_yy)
axes[0,1].set_title("sigma_yy")
#  #axes[0,1].set_ylim([2,5])
# #
axes[1,0].plot(xipoint_radii,dataline_sigma_zz)
axes[1,0].set_title("sigma_zz")
# # #axes[1,0].set_ylim([2,8])
# #
axes[1,1].plot(xipoint_radii,dataline_sigma_xy)
axes[1,1].set_title("sigma_xy")
# # #axes[1,0].set_ylim([2,8])
#
plt.savefig("stress_refined_mesh_option_"+str(refined_mesh_option)+".png")
#
#F = equationsSet.TensorInterpolateXi(iron.EquationsSetDerivedTensorTypes.DEFORMATION_GRADIENT,elementNumber, xiPosition, (3, 3))
#results['Deformation Gradient Tensor'] = F

#C = equationsSet.TensorInterpolateXi(
#    iron.EquationsSetDerivedTensorTypes.R_CAUCHY_GREEN_DEFORMATION,
#    elementNumber, xiPosition, (3, 3))
#results['Right Cauchy-Green Deformation Tensor'] = C

#E = equationsSet.TensorInterpolateXi(
#    iron.EquationsSetDerivedTensorTypes.GREEN_LAGRANGE_STRAIN,
#    elementNumber, xiPosition, (3, 3))
#results['Green-Lagrange Strain Tensor'] = E


#I1 = np.trace(C)
#I2 = 0.5 * (np.trace(C) ** 2. - np.tensordot(C, C))
#I3 = np.linalg.det(C)
#results['Invariants'] = [I1, I2, I3]

#TC = equationsSet.TensorInterpolateXi(
#    iron.EquationsSetDerivedTensorTypes.CAUCHY_STRESS,
#    elementNumber, xiPosition, (3, 3))
#results['Cauchy Stress Tensor'] = TC
#
#print(results)
# Export results
fields = iron.Fields()
fields.CreateRegion(region)
fields.NodesExport("CylinderInflation","FORTRAN")
fields.ElementsExport("CylinderInflation","FORTRAN")
fields.Finalise()
print('Program complete')