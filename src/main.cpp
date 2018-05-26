#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/Timer.h>

//#define DEBUG

using namespace std;
using Viewer = igl::viewer::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Parameter: degree of the polynomial
unsigned int polyDegree = 1;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 100; //0.1

// Parameter: grid resolution
unsigned int resolution = 30;

// Grid Expansion factor
double expansionFactor = 1.1;

//Bounding box
Eigen::RowVector3d bb_min, bb_max;
Eigen::RowVector3d extraSpace;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

// Functions
void createGrid();
void evaluateImplicitFunc();
void getLines();
bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers);

//Personal functions for other
double getDiagLength();

//Personal functions for DataStructure
void prepareQuickAccess();
int getClosest(int pointQ);
int getClosestOld(int pointQ);
std::list<int> getClosestSetOld(int pointQ, double distanceh);
Eigen::Matrix<int, Eigen::Dynamic, 1> getClosestSetConstrained(int indicePointGrid, double rayon);
Eigen::Matrix<int, Eigen::Dynamic, 1> getClosestSetConstrainedOld(int indicePointGrid, double rayon);

std::vector<int> getAdjacentsCells(unsigned int x, unsigned int y, unsigned int z);
std::vector<int> getAdjacentsCellsFromIndice(unsigned int indiceCell);
unsigned int getXCoordCellFromIndex(unsigned int index);
unsigned int getYCoordCellFromIndex(unsigned int index);
unsigned int getZCoordCellFromIndex(unsigned int index);
unsigned int getIndiceFromCoord(unsigned int x, unsigned int y, unsigned int z);

void optimizePosition();

//QuickDataStructure parameters
std::vector<std::vector<int>> quikStructure; // default construction
double dx = wendlandRadius;
double dy = wendlandRadius;
double dz = wendlandRadius;
unsigned int nbCellXEdge = 0;
unsigned int nbCellYEdge = 0;
unsigned int nbCellZEdge = 0;

//Get the indice in the QuickAcces Cell from the 3D coordinates of this cell (x,y,z)
unsigned int getIndiceFromCoord(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int numCell = (x * nbCellXEdge + y) * nbCellYEdge + z;
    if(numCell > quikStructure.size()){
        std::cerr << "Trying to access " << numCell << " but quickStructure limited to " << quikStructure.size() << endl;
        assert(numCell < quikStructure.size());
    }
    return numCell;
}

unsigned int getXCoordCellFromIndex(unsigned int index)
{
    return index / (nbCellXEdge * nbCellYEdge); //Note the integer division . This is x
}
unsigned int getYCoordCellFromIndex(unsigned int index)
{
    unsigned int x = getXCoordCellFromIndex(index);
    return (index - x * nbCellXEdge * nbCellYEdge) / nbCellYEdge; //This is y
}
unsigned int getZCoordCellFromIndex(unsigned int index)
{
    unsigned int x = getXCoordCellFromIndex(index);
    unsigned int y = getYCoordCellFromIndex(index);
    return index - x * nbCellXEdge * nbCellYEdge - y * nbCellYEdge; //This is z
}

//See :
/*
    3D {i, j, k}:
    Code C++ :	Sélectionner tout - Visualiser dans une fenêtre à part
    (i*Size_Dim_1+j)*Size_Dim_2+k
    
If you use a 3D array A[x][y][z] should be replaced by A[ x * height * depth + y * depth + z ]
If you use a 3D array A[x][y][z] should be replaced by A[ x * nbCellXEdge * nbCellYEdge + y * nbCellYEdge + z ]

If you have the 1D array A[index] and you want to see what that corresponds to in 3D,
width_index=index/(nbCellXEdge*nbCellYEdge);  //Note the integer division . This is x
height_index=(index-width_index*nbCellXEdge*nbCellYEdge)/nbCellYEdge; //This is y
depth_index=index-width_index*nbCellXEdge*nbCellYEdge- height_index*nbCellYEdge;//This is z 
*/

//See : https://stackoverflow.com/questions/33028073/c-3d-array-declaration-using-vector
//And see : http://www.cplusplus.com/forum/articles/7459/

unsigned int getIndiceFromConstraintsCoord(int xConstraintCoord, int yConstraintCoord, int zConstraintCoord)
{
    //Get X cells is belongs to
    unsigned int x = (xConstraintCoord - bb_min[0] + (extraSpace[0] / 2)) / dx; // SURE FOR 2 ?
    //Get Y cells is belongs to
    unsigned int y = (yConstraintCoord - bb_min[1] + (extraSpace[1] / 2)) / dy; // SURE FOR 1 ? // LES VALEURS PEUVENT ETRE NEGATIVES
    //Get Z cells is belongs to
    unsigned int z = (zConstraintCoord - bb_min[2] + (extraSpace[2] / 2)) / dz; //SURE FOR 0 ?

    //Put it in the right cell
    unsigned int coordCell = getIndiceFromCoord(x, y, z); // (x * nbCellXEdge + y) * nbCellYEdge + z;
#ifdef DEBUG
    std::cout << " Point : x=" << constrained_points.row(i)[0] << " ,y=" << constrained_points.row(i)[1] << " ,z=" << constrained_points.row(i)[2] << " attributed to ";
    std::cout << " cell : x=" << x << " ,y=" << y << " ,z=" << z << ". sizeCell = " << x << " " << y << " " << z << endl;
#endif
    return coordCell;
}

//Give the list of index of neighboors cells from the currentCell parameter (1 distance, 27 cells returns usually)

std::vector<int> getAdjacentsCellsFromIndice(unsigned int index)
{
    int x = index / (nbCellXEdge * nbCellYEdge);
    int y = (index - x * nbCellXEdge * nbCellYEdge) / nbCellYEdge;
    int z = index - x * nbCellXEdge * nbCellYEdge - y * nbCellYEdge;

    return getAdjacentsCells(x, y, z);
}

std::vector<int> getAdjacentsCells(unsigned int x, unsigned int y, unsigned int z)
{
    std::vector<int> listAdjacentsCells = {};

    int nbCasesVoisines = 0;

    for (int i = -1; i < 2; i++)
    {
        if (x + i < nbCellXEdge && x + i >= 0)
        {
            for (int j = -1; j < 2; j++)
            {
                if (y + j < nbCellYEdge && y + j >= 0)
                {
                    for (int k = -1; k < 2; k++)
                    {
                        if (z + k < nbCellZEdge && z + k >= 0)
                        {
                            listAdjacentsCells.push_back(getIndiceFromCoord(x + i, y + j, z + k));
                            nbCasesVoisines++;
                        }
                    }
                }
            }
        }
    }

//DEBUG //
#ifdef DEBUG
    std::cout << "Nombre de cases voisines de : x=" << x << " y=" << y << " z=" << z << " est " << nbCasesVoisines << endl;
#endif
    return listAdjacentsCells;
}

//Initialize data structure
void prepareQuickAccess()
{
    // Grid bounds: axis-aligned bounding box
    bb_min = constrained_points.colwise().minCoeff(); // NOTE : SHould it be on ConstrainedPoint ?
    bb_max = constrained_points.colwise().maxCoeff();

    // Bounding box dimensions
    Eigen::RowVector3d dim = (bb_max - bb_min) * expansionFactor;
    extraSpace = dim - (bb_max - bb_min); //We example it a little, to "englob" the mesh correctly

    // Grid spacing = we want cells of Wendland radius size
    dx = wendlandRadius; //(double)(resolution - 1);
    dy = wendlandRadius; //(double)(resolution - 1);
    dz = wendlandRadius; //(double)(resolution - 1);

    unsigned int defaultNbVoisin = 0;
    //unsigned int nbCellOneEdge = dim[0];// PREVIOUSLY : (resolution - 1);
    //We are creating +1 cell because the wendland cells are not aligned with the grid
    nbCellXEdge = ((dim[0] + extraSpace[0]) / wendlandRadius) + 1;
    nbCellYEdge = ((dim[1] + extraSpace[1]) / wendlandRadius) + 1;
    nbCellZEdge = ((dim[2] + extraSpace[2]) / wendlandRadius) + 1;

    //    unsigned int numCell = (x * nbCellXEdge + y) * nbCellYEdge + z;
    unsigned int sizeQuickStructure = ((nbCellXEdge * nbCellXEdge + nbCellYEdge) * nbCellYEdge) + nbCellZEdge + 1; // Quickfix for relation 3D -> 1D. Unknown reason, we can't get the value 230, for example
    std::cout << "sizeQuickStructure" << sizeQuickStructure << endl;
    //unsigned int sizeQuickStructure = (nbCellXEdge * nbCellYEdge * nbCellZEdge)*1.1; // Quickfix for relation 3D -> 1D. Unknown reason, we can't get the value 230, for example
    //std::vector<std::vector<std::vector<double>>> Array3D(X, std::vector<std::vector<double>>(Y, std::vector<double>(Z)));
    //quikStructure = std::vector<std::vector<int> sizeQuickStructure>; // default construction

    //Resize the data structure to the good size / clean up the datastructure
    quikStructure.resize(sizeQuickStructure);
    for (int i = 0; i < sizeQuickStructure; ++i)
    {
        quikStructure[i].resize(defaultNbVoisin);
    }

    /*for (int i = 0; i < sizeQuickStructure; i++)
    {
        for (int j = 0; j < sizeQuickStructure; j++)
        {
            for (int k = 0; k < sizeQuickStructure; k++)
            {
                int indiceCurr = getIndiceFromCoord(i,j,k);
                quikStructure[indiceCurr].resize(defaultNbVoisin);
            }
        }
    }*/

    //For all points
    unsigned int coordCell = 0;
    for (int i = 0; i < constrained_points.rows(); i++)
    {
        coordCell = getIndiceFromConstraintsCoord(constrained_points.row(i)[0], constrained_points.row(i)[1], constrained_points.row(i)[2]);
        quikStructure[coordCell].push_back(i);
    }

//DEBUG CODE
#ifdef DEBUG

    unsigned int nbPointAdded = 0;
    for (unsigned int i = 0; i < quikStructure.size(); i++)
    {

        std::cout << " case : " << i << " values : ";

        for (unsigned int j = 0; j < quikStructure[i].size(); j++)
        {
            std::cout << quikStructure[i][j] << " ";
            nbPointAdded++;
        }

        std::cout << endl;
    }

    std::cout << " NbPointsAdded " << nbPointAdded << endl;
    std::cout << "constrainedPoints " << constrained_points.rows() << endl;
#endif
}

// PointQ Should be part of the cloudPoint P
int getClosest(int pointQ)
{
    //Sanity check
    assert(pointQ < P.rows());

    Eigen::RowVector3d curDistance;
    Eigen::MatrixXd refCoords = P.row(pointQ);

    //Get the cell in which the P dot is.
    unsigned int coordCell = getIndiceFromConstraintsCoord(refCoords.row(0)[0], refCoords.row(0)[1], refCoords.row(0)[2]);

    //Initialisation (Just get random value to start)
    int nearestNeighboor = (pointQ + 1) % P.rows(); //Prevent problem if i = pointQ
    curDistance = refCoords.row(0) - P.row(nearestNeighboor);
    double bestDistanceVal = curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2];
    double curDistanceVal = bestDistanceVal + 1; //Just to make it different from the best one

    //Is there points in the cell of the reference Point ?
    if (quikStructure[coordCell].size() != 0)
    {
        //Cas 1 : Neighboors in his own cell
        for (int i = 0; i < quikStructure[coordCell].size(); i++)
        {
            //For each neighboor in his cell
            assert(quikStructure[coordCell][i] < P.rows());

            //Calcul distance
            curDistance = refCoords.row(0) - P.row(quikStructure[coordCell][i]);
            curDistanceVal = curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2];
            //Note : we don't sqrt() it to save some calculations. CONSISTENT WITH COMPARISON VALUE !

            //Compare (if not with itself)
            if (bestDistanceVal > curDistanceVal && pointQ != quikStructure[coordCell][i])
            {
                //We found a best value
                nearestNeighboor = quikStructure[coordCell][i];
                bestDistanceVal = curDistanceVal;
            }
        }
    }
    else
    {
        //Cas 2 : No neighboors in his cell, look around
        std::vector<int> cellsToVisit = getAdjacentsCellsFromIndice(coordCell);
        bool aLeastANeighboor = false;

        //We visit each neighboor cell
        for (int c = 0; c < cellsToVisit.size(); c++)
        {
            //We look the neighboors in the neighboors cells
            for (int i = 0; i < quikStructure[cellsToVisit[c]].size(); i++)
            {
                assert(quikStructure[cellsToVisit[c]][i] < P.rows());

                //Calcul distance
                curDistance = refCoords.row(0) - P.row(quikStructure[cellsToVisit[c]][i]);
                curDistanceVal = curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2];
                //Note : we don't sqrt() it to save some calculations. CONSISTENT WITH COMPARISON VALUE !

                //Compare (if not with itself)
                if (bestDistanceVal > curDistanceVal && pointQ != quikStructure[cellsToVisit[c]][i])
                {
                    aLeastANeighboor = true;
                    //We found a best value
                    nearestNeighboor = quikStructure[cellsToVisit[c]][i];
                    bestDistanceVal = curDistanceVal;
                }
            }
        }

        // else {//We look everywhere}
        if (aLeastANeighboor == false)
        {
            nearestNeighboor = getClosestOld(pointQ);
        }
    }

    return nearestNeighboor;
}

int getClosestOld(int pointQ)
{ // OR const Eigen::MatrixXd& refPoint, int indiceRefPoint
    //Sanity check
    assert(pointQ < P.size());

    Eigen::RowVector3d curDistance;
    Eigen::MatrixXd refCoords = P.row(pointQ);

    //Initialisation
    int nearestNeighboor = (pointQ + 1) % P.rows(); //Prevent problem if i = pointQ
    curDistance = refCoords.row(0) - P.row(nearestNeighboor);
    double bestDistanceVal = curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2];
    double curDistanceVal = bestDistanceVal + 1; //Just to make it different from the best one

    for (int i = 0; i < P.rows(); i++)
    {
        //Calcul distance
        curDistance = refCoords.row(0) - P.row(i);
        curDistanceVal = curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2];
        //Note : we don't sqrt() it to save some calculation. CONSISTENT WITH COMPARISON VALUE !

        //Compare (if not with itself)
        if (bestDistanceVal > curDistanceVal && pointQ != i)
        {
            //We found a best value
            nearestNeighboor = i;
            bestDistanceVal = curDistanceVal;
        }
    }

    return nearestNeighboor;
}

std::list<int> getClosestSetOld(int pointQ, double distanceh)
{
    //Sanity check
    assert(pointQ < P.size());

    std::list<int> closestPointsList = {};

    //Initialisation
    Eigen::RowVector3d curDistance;
    Eigen::MatrixXd refCoords = P.row(pointQ);
    double curDistanceVal = 0;

    for (int i = 0; i < P.rows(); i++)
    {
        //Calcul distance with current dot
        curDistance = refCoords.row(0) - P.row(i);
        curDistanceVal = sqrt(curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2]);
        //Note : we HAVE sqrt() it to save some calculation. CONSISTENT WITH COMPARISON VALUE !

        //Compare (if not with itself)
        if (distanceh > curDistanceVal && pointQ != i)
        {
            //We found a good value, we add it to the list
            closestPointsList.push_back(i);
        }
    }

    return closestPointsList;
}

Eigen::Matrix<int, Eigen::Dynamic, 1> getClosestSetConstrained(int indicePointGrid, double rayon)
{
    //Sanity check
    assert(indicePointGrid < grid_points.rows());

    //At maximum, this point can have #constrainedPoint neighboor
    Eigen::Matrix<int, Eigen::Dynamic, 1> closestPointsList(constrained_points.rows(), 1);

    //Initialisation
    Eigen::RowVector3d curDistance;                               //The current distance (vector) between the gridpoint and the neighboor
    Eigen::MatrixXd refCoords = grid_points.row(indicePointGrid); // The coordinates of the grid point

    // Grid bounds: axis-aligned bounding box
    bb_min = constrained_points.colwise().minCoeff();
    bb_max = constrained_points.colwise().maxCoeff();

    double curDistanceVal = 0; //The current calculated distance between the gridpoint and the neighboor
    int nbNeighboor = 0;       //The N° of the current neigbhoor processed.

    //Get the cell in which the current grid dot is.
    unsigned int coordCell = getIndiceFromConstraintsCoord(refCoords.row(0)[0], refCoords.row(0)[1], refCoords.row(0)[2]);

    //Gt neighboors cells
    std::vector<int> cellsToVisit = getAdjacentsCellsFromIndice(coordCell);

    //We visit each neighboor cell
    for (int c = 0; c < cellsToVisit.size(); c++)
    {
        //We look the neighboors in the neighboors cells
        for (int i = 0; i < quikStructure[cellsToVisit[c]].size(); i++)
        {
            assert(quikStructure[cellsToVisit[c]][i] < constrained_points.rows());
            //Calcul distance with current dot
            curDistance = refCoords.row(0) - constrained_points.row(quikStructure[cellsToVisit[c]][i]);
            curDistanceVal = sqrt(curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2]);

            //Compare
            if (rayon > curDistanceVal)
            {
                //We found a good value, we add it to the list
                closestPointsList.row(nbNeighboor) << quikStructure[cellsToVisit[c]][i];
                nbNeighboor++;
            }
        }
    }

    if (nbNeighboor == 0)
    {
        Eigen::Matrix<int, Eigen::Dynamic, 1> returnVide(0, 1);
        return returnVide;
    }
    else
    {
        closestPointsList.conservativeResize(nbNeighboor, Eigen::NoChange);
        return closestPointsList;
    }
}

Eigen::Matrix<int, Eigen::Dynamic, 1> getClosestSetConstrainedOld(int indicePointGrid, double rayon)
{
    //Sanity check
    assert(indicePointGrid < grid_points.rows());

    //At maximum, this point can have #constrainedPoint neighboor
    Eigen::Matrix<int, Eigen::Dynamic, 1> closestPointsList(constrained_points.rows(), 1);

    //Initialisation
    Eigen::RowVector3d curDistance; //The current distance (vector) between the gridpoint and the neighboor
    double curDistanceVal = 0;      //The current calculated distance between the gridpoint and the neighboor

    Eigen::MatrixXd refCoords = grid_points.row(indicePointGrid); // The coordinates of the grid point
    int nbNeighboor = 0;                                          //The N° of the current neigbhoor processed.

    //For all possibles neighboors
    for (int i = 0; i < constrained_points.rows(); i++)
    {
        //Calcul distance with current dot
        curDistance = refCoords.row(0) - constrained_points.row(i);
        curDistanceVal = sqrt(curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2]);

        //Compare (if not with itself)
        if (rayon > curDistanceVal)
        {
            //We found a good value, we add it to the list
            closestPointsList.row(nbNeighboor) << i;
            nbNeighboor++;
        }
    }

    if (nbNeighboor == 0)
    {
        Eigen::Matrix<int, Eigen::Dynamic, 1> returnVide(0, 1);
        return returnVide;
    }
    else
    {
        closestPointsList.conservativeResize(nbNeighboor, Eigen::NoChange);
        return closestPointsList;
    }
}

// Creates a grid_points array for the simple sphere example. The points are
// stacked into a single matrix, ordered first in the x, then in the y and
// then in the z direction. If you find it necessary, replace this with your own
// function for creating the grid.
void createGrid()
{
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines.resize(0, 6);
    grid_values.resize(0);
    V.resize(0, 3);
    F.resize(0, 3);
    FN.resize(0, 3);

    optimizePosition();

    // Grid bounds: axis-aligned bounding box
    bb_min = constrained_points.colwise().minCoeff();
    bb_max = constrained_points.colwise().maxCoeff();

    // Bounding box dimensions
    Eigen::RowVector3d dim = (bb_max - bb_min) * expansionFactor; //We example it a little, to "englob" the mesh correctly
    Eigen::RowVector3d extraSpace = dim - (bb_max - bb_min);      //We example it a little, to "englob" the mesh correctly

    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);

    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);

    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x)
    {
        for (unsigned int y = 0; y < resolution; ++y)
        {
            for (unsigned int z = 0; z < resolution; ++z)
            {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min - (extraSpace / 2) + Eigen::RowVector3d(x * dx, y * dy, z * dz);
                // We move it from half of the extraSpace generated
            }
        }
    }
}

// Function for explicitly evaluating the implicit function at the grid points using MLS
void evaluateImplicitFunc()
{
    igl::Timer time;
    time.start();
    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    //We calculate the nb of "basis" in a row, depending on the chosen polydegree
    long currentIndex = 0;
    long nbBases = 0;
    switch (polyDegree)
    {
    case 0:
        nbBases = 1;
        break;
    case 1:
        nbBases = 4;
        break;
    case 2:
        nbBases = 10;
        break;
    default: // BAD PARAMETER !
        assert(false);
        break;
    }

    //Iter through all gridpoints
    for (unsigned int x = 0; x < resolution; ++x)
    {
        for (unsigned int y = 0; y < resolution; ++y)
        {
            for (unsigned int z = 0; z < resolution; ++z)
            {

                //We consider a point of the grid
                // grid_points.row(currentIndex);

                //Compute the futur basis
                double xCoordGrid = grid_points.row(currentIndex)[0]; //2 ?
                double yCoordGrid = grid_points.row(currentIndex)[1]; //1 ?
                double zCoordGrid = grid_points.row(currentIndex)[2]; //0 ?

                Eigen::MatrixXd baseGrid = Eigen::MatrixXd::Zero(1, nbBases);
                switch (polyDegree)
                {
                //VOLONTARY NO BREAK TO PREVENT CODE DUPLICATION
                case 2:
                    baseGrid.row(0)[9] = xCoordGrid * zCoordGrid;
                    baseGrid.row(0)[8] = yCoordGrid * zCoordGrid;
                    baseGrid.row(0)[7] = xCoordGrid * yCoordGrid;
                    baseGrid.row(0)[6] = zCoordGrid * zCoordGrid;
                    baseGrid.row(0)[5] = yCoordGrid * yCoordGrid;
                    baseGrid.row(0)[4] = xCoordGrid * xCoordGrid;

                case 1:
                    baseGrid.row(0)[3] = zCoordGrid;
                    baseGrid.row(0)[2] = yCoordGrid;
                    baseGrid.row(0)[1] = xCoordGrid;

                case 0:
                    baseGrid.row(0)[0] = 1;
                }

                //We get his neighboorhood (in the wenderradius)
                Eigen::Matrix<int, Eigen::Dynamic, 1> resultSet = getClosestSetConstrained(currentIndex, wendlandRadius);

                //We create the necessary dataStructures
                Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(resultSet.rows(), resultSet.rows());
                Eigen::MatrixXd base = Eigen::MatrixXd::Zero(resultSet.rows(), nbBases);
                Eigen::MatrixXd coefs = Eigen::MatrixXd::Zero(nbBases, 1);
                Eigen::MatrixXd constraintsVal = Eigen::MatrixXd::Zero(resultSet.rows(), 1);

                if (resultSet.rows() >= nbBases)
                { //The current point of the grid has some neighboors

                    //For each point of the neighboorhood :
                    for (int i = 0; i < resultSet.rows(); i++)
                    {
                        //We get the id of this points (ID valid in contrainted_points, constrained_value, etc.)
                        long idNeighboorPoint = resultSet.row(i)[0];

                        //We get the constraint of this point & We store the constraint
                        double constValue = constrained_values.row(idNeighboorPoint)[0];
                        constraintsVal.row(i)[0] = constValue;

                        //We get the coordinates of this point
                        Eigen::MatrixXd currentCoords(1, 3);
                        currentCoords.row(0) << constrained_points.row(idNeighboorPoint);

                        //Calcul distance between actualPoint and neighboor
                        Eigen::RowVector3d curDistance = grid_points.row(currentIndex) - currentCoords.row(0);
                        double curDistanceVal = sqrt(curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2]);

                        //We calcul the wendeland weight factor for this point = Weight of this neighboor in the equation
                        double wendlandValue = pow((1 - (curDistanceVal / wendlandRadius)), 4) * ((4 * curDistanceVal / wendlandRadius) + 1);

                        //We store the weight, the coordinates, the weight and the constraints
                        weights.row(i)[i] = wendlandValue; //i - i because it's diagonal matrix

                        //Calculate the bases values according to the degree selected
                        switch (polyDegree)
                        {
                        //VOLONTARY NO BREAK TO PREVENT CODE DUPLICATION
                        case 2:
                            base.row(i)[9] = xCoordGrid * zCoordGrid;
                            base.row(i)[8] = yCoordGrid * zCoordGrid;
                            base.row(i)[7] = xCoordGrid * yCoordGrid;
                            base.row(i)[6] = zCoordGrid * zCoordGrid;
                            base.row(i)[5] = yCoordGrid * yCoordGrid;
                            base.row(i)[4] = xCoordGrid * xCoordGrid;

                        case 1:
                            base.row(i)[3] = zCoordGrid;
                            base.row(i)[2] = yCoordGrid;
                            base.row(i)[1] = xCoordGrid;

                        case 0:
                            base.row(i)[0] = 1;
                        }
                    }

                    //DEBUG // SANITY CHECK
                    //We ensure there is no "0-only" row in the weight matrix.
                    /*for (int k = 0; k < weights.rows(); k++)
                    {
                        double maxTMP = weights.row(k)[k];
                        if (maxTMP == 0)
                        {
                            //DEBUG // std::cout << "On row " << k << " is nul "<< endl;
                            assert(maxTMP != 0);
                        }
                    }*/

                    //We solve the system (Eigen call)
                    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(resultSet.rows(), nbBases);
                    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(resultSet.rows(), 1);
                    //Eigen::VectorXd coefs = Eigen::VectorXd::Zero(nbBases,1);

                    A = weights * base;
                    B = weights * constraintsVal;

#ifdef DEBUG
                    std::cout << "Solving the system : " << endl;
                    for (int l = 0; l < weights.rows(); l++)
                    {
                        std::cout << "weight : " << weights.row(l)[l] << " \tbasis :  ";
                        for (int k = 0; k < nbBases; k++)
                        {
                            std::cout << base.row(l)[k] << " ";
                        }
                        std::cout << " = ";
                        std::cout << "\tweight : " << weights.row(l)[l] << " \tconstraint :  " << constraintsVal.row(l)[0] << endl;
                    }
#endif

                    Eigen::VectorXd coefs = A.colPivHouseholderQr().solve(B);
                    //Eigen::VectorXd coefs = A.jacobiSvd().solve(B); //PROBLEM ICI !

#ifdef DEBUG
                    std::cout << "Coefficients : " << endl;
                    for (int k = 0; k < nbBases; k++)
                    {
                        std::cout << coefs[k] << " ";
                    }

                    std::cout << " ------ " << endl;
// SEE : https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
#endif

                    //We compute the value of the function for this particular point
                    double functionValue = 0;
                    for (int j = 0; j < nbBases; j++)
                    {
                        functionValue += coefs[j] * baseGrid.row(0)[j]; // We do coef[0]*1 + coef[1]*x + coef[2]*y + ...
                    }

                    //We store the value
                    //grid_values[currentIndex] = grid_points.row(index) ...;
                    grid_values[currentIndex] = functionValue;
#ifdef DEBUG
                    std::cout << " Value calculated : " << functionValue << endl;
#endif
                    //We color the point : NO = IT'S AFTER
                }
                else
                {
                    //The current point of the grid has NO neighboors
                    grid_values[currentIndex] = 100;
                    //We store an arbitrary high value
                }

                currentIndex++;
            }
        }
    }
    
    time.stop();
    std::cout << "Sampling time : " <<  time.getElapsedTime() << endl;
}

void optimizePosition()
{
    // ## PCA computation :

    //Minimise the square error of approximating data and centers the data
    Eigen::MatrixXd centered = P.rowwise() - P.colwise().mean();
    //Calculate the covariance matrix (3*3)
    Eigen::MatrixXd cov = centered.adjoint() * centered;

    //Prepare the calculations for eigen vectors and eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov);

    //Get the Eigenvectors
    Eigen::Vector3f eigenA = eigensolver.eigenvectors().col(0).normalized().cast<float>();
    Eigen::Vector3f eigenB = eigensolver.eigenvectors().col(1).normalized().cast<float>();
    Eigen::Vector3f eigenC = eigensolver.eigenvectors().col(2).normalized().cast<float>();
    Eigen::Vector3f eigenVal = eigensolver.eigenvalues().cast<float>();

    Eigen::Vector3f target(1.0, 0.0, 0.0);

#ifdef DEBUG
    std::cout << "EigenVector 1 : " << eigenA[0] << " " << eigenA[1] << " " << eigenA[2] << " of eigenvalue : " << eigenVal[0] << endl;
    std::cout << "EigenVector 2 : " << eigenB[0] << " " << eigenB[1] << " " << eigenB[2] << " of eigenvalue : " << eigenVal[1] << endl;
    std::cout << "EigenVector 3 : " << eigenC[0] << " " << eigenC[1] << " " << eigenC[2] << " of eigenvalue : " << eigenVal[2] << endl;
    //# Note : eigenvectors are sorted as the eigenvalues, wich are sorted ascending order. Max is the third one.
#endif

    // ## Rotation preparation :
    Eigen::Matrix3f R = Eigen::Quaternionf().setFromTwoVectors(eigenC, target).toRotationMatrix().cast<float>();

#ifdef DEBUG
    std::cout << "Rotation matrix : " << endl;
    std::cout << R.row(0)[0] << " " << R.row(0)[1] << " " << R.row(0)[2] << endl;
    std::cout << R.row(1)[0] << " " << R.row(1)[1] << " " << R.row(1)[2] << endl;
    std::cout << R.row(2)[0] << " " << R.row(2)[1] << " " << R.row(2)[2] << endl;
#endif



    // ## Rotation computation :
    //vec2 = gen1 * vec1;
    Eigen::MatrixXd constrained_pointsTMP = Eigen::MatrixXd::Zero(constrained_points.rows(), 3);
    Eigen::MatrixXd PTMP = Eigen::MatrixXd::Zero(P.rows(), 3);
    Eigen::MatrixXd NTMP = Eigen::MatrixXd::Zero(N.rows(), 3);

    //constrained_pointsTMP = constrained_points.transpose();
    //constrained_pointsTMP = R.linear()*constrained_pointsTMP;

    //
    Eigen::Matrix3d normalMatrix = R.inverse().transpose().cast<double>();
    //n2 = (normalMatrix * n1).normalized();

    for (int i = 0; i < N.rows(); i++)
    {
        Eigen::Vector3d norTMP;
        Eigen::Vector3d norOUT;

        norTMP = N.row(i).transpose();

#ifdef DEBUG
        std::cout << "Current normal vector (inital) : " << norTMP[0] << " " << norTMP[1] << " " << norTMP[2] << endl;
#endif

        //norOUT = R.cast<double>() * norTMP;
        norOUT = (normalMatrix * norTMP); // .normalize()

        NTMP.row(i)[0] = norOUT[0];
        NTMP.row(i)[1] = norOUT[1];
        NTMP.row(i)[2] = norOUT[2];

#ifdef DEBUG
        std::cout << "Current normal vector (rotation) : " << norOUT[0] << " " << norOUT[1] << " " << norOUT[2] << endl;
#endif
    }

    for (int i = 0; i < constrained_points.rows(); i++)
    {
        Eigen::Vector3d vecTMP;
        Eigen::Vector3d vecOUT;

        vecTMP = constrained_points.row(i).transpose();

#ifdef DEBUG
        std::cout << "Current vector (inital) : " << vecTMP[0] << " " << vecTMP[1] << " " << vecTMP[2] << endl;
#endif

        vecOUT = R.cast<double>() * vecTMP;

        constrained_pointsTMP.row(i)[0] = vecOUT[0];
        constrained_pointsTMP.row(i)[1] = vecOUT[1];
        constrained_pointsTMP.row(i)[2] = vecOUT[2];

#ifdef DEBUG
        std::cout << "Current vector (rotation) : " << vecOUT[0] << " " << vecOUT[1] << " " << vecOUT[2] << endl;
#endif
    }

    for (int i = 0; i < P.rows(); i++)
    {
        Eigen::Vector3d vecTMPP;
        Eigen::Vector3d vecOUTP;

        vecTMPP = P.row(i).transpose();

#ifdef DEBUG
        std::cout << "Current vector (inital) : " << vecTMPP[0] << " " << vecTMPP[1] << " " << vecTMPP[2] << endl;
#endif

        vecOUTP = R.cast<double>() * vecTMPP;

        PTMP.row(i)[0] = vecOUTP[0];
        PTMP.row(i)[1] = vecOUTP[1];
        PTMP.row(i)[2] = vecOUTP[2];
#ifdef DEBUG
        std::cout << "Current vector (rotation) : " << vecOUTP[0] << " " << vecOUTP[1] << " " << vecOUTP[2] << endl;
#endif
    }
    //constrained_pointsTMP *= R;
    constrained_points = constrained_pointsTMP;
    P = PTMP;
    N = NTMP;

    //Recalculate necessary values
    bb_min = constrained_points.colwise().minCoeff(); // NOTE : SHould it be on ConstrainedPoint ?
    bb_max = constrained_points.colwise().maxCoeff();

    //Eigen::Matrix3f R; //Give the rotation from vector A to vector B

}

// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines()
{
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x < resolution; ++x)
    {
        for (unsigned int y = 0; y < resolution; ++y)
        {
            for (unsigned int z = 0; z < resolution; ++z)
            {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1)
                {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1)
                {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1)
                {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}

bool isNearestNeighboor(const Eigen::MatrixXd &refPoint, int indiceRefPoint, const Eigen::MatrixXd &distantPoint)
{
    Eigen::RowVector3d currentDist = refPoint - distantPoint;
    double currentDistVal = currentDist[0] * currentDist[0] + currentDist[1] * currentDist[1] + currentDist[2] * currentDist[2];
    //Note : we don't sqrt() it to save some calculation

    //We verify that each other point is not nearer than the actual distant point
    Eigen::RowVector3d currentComparableDist;
    double currentComparableDistVal;

    for (int i = 0; i < P.rows(); i++)
    {
        //Calcul distance
        currentComparableDist = refPoint.row(0) - P.row(i);
        currentComparableDistVal = currentComparableDist[0] * currentComparableDist[0] + currentComparableDist[1] * currentComparableDist[1] + currentComparableDist[2] * currentComparableDist[2];
        //Note : we don't sqrt() it to save some calculation. CONSISTENT WITH COMPARISON VALUE !

        //Compare (if not with itself)
        if (currentDistVal > currentComparableDistVal && indiceRefPoint != i)
        {
            return false;
        }
    }

    return true;
}

double getDiagLength()
{
    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();

    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;
    double diag = sqrt(dim[0] * dim[0] + dim[1] * dim[1] + dim[2] * dim[2]);

    return diag;
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers)
{
    if (key == '1')
    {
        // Show imported points
        viewer.data.clear();
        viewer.core.align_camera_center(P);
        viewer.data.point_size = 11;
        viewer.data.add_points(P, Eigen::RowVector3d(0, 0, 0));
    }

    if (key == '2')
    {
        // Show all constraints
        viewer.data.clear();
        viewer.core.align_camera_center(P);
        // Add your code for computing auxiliary constraint points here
        Eigen::MatrixXd currentNormal(1, 3);
        Eigen::MatrixXd currentPepsPlus(1, 3);
        Eigen::MatrixXd currentPepsMinus(1, 3);

        double diag = getDiagLength();

        double globalEps = 0.01 * diag;

        //USEFUL ? N.rowwise().normalize();
        //Resize of the constrained_points & constrained_values
        constrained_points = Eigen::MatrixXd::Zero(N.rows() * 3, 3);
        constrained_values = Eigen::VectorXd::Zero(N.rows() * 3, 1);

        for (int i = 0; i < N.rows(); i++)
        {
            currentNormal.row(0) = N.row(i);

#ifdef DEBUG
            std::cout << "current normal : " << currentNormal.row(0) << endl;
#endif

            double localEspPlus = globalEps;
            Eigen::MatrixXd pointEspPlus;

            //Calculus of the P eps+
            pointEspPlus = P.row(i) + localEspPlus * currentNormal.row(0);
            while (!isNearestNeighboor(P.row(i), i, pointEspPlus))
            {
                localEspPlus = localEspPlus / 2;
                pointEspPlus = P.row(i) + localEspPlus * currentNormal.row(0);
            }

            double localEspMinus = -globalEps;
            Eigen::MatrixXd pointEspMinus;

            //Calculus of the P eps-
            pointEspMinus = P.row(i) + localEspMinus * currentNormal.row(0);
            while (!isNearestNeighboor(P.row(i), i, pointEspMinus))
            {
                localEspMinus = localEspMinus / 2;
                pointEspMinus = P.row(i) + localEspMinus * currentNormal.row(0);
            }
#ifdef DEBUG
            std::cout << " localEspMinus " << localEspMinus << "\t localEspPlus " << localEspPlus << endl;
#endif
            //Add the point in the data structure
            constrained_points.row(i) << P.row(i);
            constrained_points.row(i + N.rows()) << pointEspPlus.row(0);
            constrained_points.row(i + 2 * N.rows()) << pointEspMinus.row(0);

            //Add the constraint value for each point
            constrained_values.row(i)[0] = 0;
            constrained_values.row(i + N.rows())[0] = localEspPlus;
            constrained_values.row(i + 2 * N.rows())[0] = localEspMinus;

#ifdef DEBUG
            std::cout << "Add a constrained point : \t" << constrained_points.row(i)[0] << " " << constrained_points.row(i)[1] << " " << constrained_points.row(i)[2] << endl;
            std::cout << "Of constrained value : \t" << constrained_values.row(i)[0] << endl;
            std::cout << "Add a constrained point : \t" << constrained_points.row(i + N.rows()) << " " << constrained_points.row(i)[1] << " " << constrained_points.row(i)[2] << endl;
            std::cout << "Of constrained value + : \t" << constrained_values.row(i + N.rows())[0] << endl;
            std::cout << "Add a constrained point : \t" << constrained_points.row(i + 2 * N.rows())[0] << " " << constrained_points.row(i)[1] << " " << constrained_points.row(i)[2] << endl;
            std::cout << "Of constrained value - : \t" << constrained_values.row(i + 2 * N.rows())[0] << endl;
#endif
        }

        // Add code for displaying all points, as above
        viewer.data.clear();
        viewer.core.align_camera_center(P);

        viewer.data.point_size = 11;
        viewer.data.add_points(constrained_points.topRows(N.rows()), Eigen::RowVector3d(0, 0, 0));
        //See : https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html

        viewer.data.point_size = 8;
        //Middle class set of points = Exterior points = +Epsilon
        viewer.data.add_points(constrained_points.topRows(N.rows() * 2).bottomRows(N.rows()), Eigen::RowVector3d(255, 0, 0));
        //Last class set of points = interieur points = -Epsilon
        viewer.data.add_points(constrained_points.bottomRows(N.rows()), Eigen::RowVector3d(255, 255, 0));
    }

    if (key == '3')
    {
        // Show grid points with colored nodes and connected with lines
        viewer.data.clear();
        viewer.core.align_camera_center(P);


        /*** begin: sphere example, replace (at least partially) with your code ***/
        // Add code for creating a grid
        // Make grid
        createGrid();

        //Prepare the quick access data structure
        prepareQuickAccess();

        // Add your code for evaluating the implicit function at the grid points
        // Evaluate implicit function
        evaluateImplicitFunc();

        // Add code for displaying points and lines
        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i)
        {
            double value = grid_values(i);
            if (value < 0)
            {
                grid_colors(i, 1) = 1;
            }
            else
            {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.data.point_size = 8;
        viewer.data.add_points(grid_points, grid_colors);
        viewer.data.add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                              grid_lines.block(0, 3, grid_lines.rows(), 3),
                              Eigen::RowVector3d(0.8, 0.8, 0.8));
        /*** end: sphere example ***/
    }

    if (key == '4')
    {
        // Show reconstructed mesh
        viewer.data.clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0))
        {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0)
        {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        igl::per_face_normals(V, F, FN);
        viewer.data.set_mesh(V, F);
        viewer.data.show_lines = true;
        viewer.data.show_faces = true;
        viewer.data.set_normals(FN);
    }

    if (key == '6')
    {
        optimizePosition();

        // Add code for displaying all points, as above
        viewer.data.clear();
        viewer.core.align_camera_center(P);

        viewer.data.point_size = 11;
        viewer.data.add_points(constrained_points.topRows(N.rows()), Eigen::RowVector3d(0, 0, 0));
        //See : https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html

        viewer.data.point_size = 8;
        //Middle class set of points = Exterior points = +Epsilon
        viewer.data.add_points(constrained_points.topRows(N.rows() * 2).bottomRows(N.rows()), Eigen::RowVector3d(255, 0, 0));
        //Last class set of points = interieur points = -Epsilon
        viewer.data.add_points(constrained_points.bottomRows(N.rows()), Eigen::RowVector3d(255, 255, 0));
    }

    //Auto-test functions
    if (key == '7')
    {
        std::cout << "TEST OF 'GETCLOSEST' ... " << endl;
        for (int i = 0; i < P.rows(); i++)
        {
            int val = getClosest(i);
            int val2 = getClosestOld(i);
            bool isTrue = isNearestNeighboor(P.row(i), i, P.row(val));
            assert(isTrue);
            assert(val == val2);
        }
        std::cout << "getClosest TEST : PASSED (verfiied with 'isNearestNeighboor')" << endl;
    }

    if (key == '8')
    {
        prepareQuickAccess();

        std::cout << "TEST OF 'getClosestSetOld' ... " << endl;
        double diag = getDiagLength();

        for (int i = 0; i < P.rows(); i++)
        {
            std::list<int> resultSet = getClosestSetOld(i, diag * 0.1);
            std::cout << "Size of the list for " << i << " is : " << resultSet.size() << endl;
        }
        std::cout << "Please manually verify the passage of this test." << endl;
    }

    if (key == '9')
    {
        prepareQuickAccess();
        double diag = getDiagLength();
        createGrid();

        std::cout << "TEST OF 'getClosestSetConstrained' ... " << endl;

        Eigen::Matrix<int, Eigen::Dynamic, 1> resultSet;
        Eigen::Matrix<int, Eigen::Dynamic, 1> resultSetOld;

        for (int i = 0; i < grid_points.rows(); i++)
        {
            resultSet = getClosestSetConstrained(i, wendlandRadius);
            resultSetOld = getClosestSetConstrainedOld(i, wendlandRadius);

            std::cout << "Size of the list for " << i << " is : " << resultSet.rows() << " old tech : " << resultSetOld.rows() << endl;
            assert(resultSet.rows() == resultSetOld.rows());
        }
    }

    return true;
}

bool callback_load_mesh(Viewer &viewer, string filename)
{
    igl::readOFF(filename, P, F, N);
    callback_key_down(viewer, '1', 0);
    return true;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage ex2_bin <mesh.off>" << endl;
        igl::readOFF("../data/sphere.off", P, F, N);
    }
    else
    {
        // Read points and normals
        igl::readOFF(argv[1], P, F, N);
    }

    Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    //viewer.callback_load_mesh = callback_load_mesh;

    viewer.callback_init = [&](Viewer &v) {
        // Add widgets to the sidebar.
        v.ngui->addButton("Load Points", [&]() {
            std::string fname = igl::file_dialog_open();
            if (fname.length() == 0)
                return;
            callback_load_mesh(v, fname);
        });

        v.ngui->addGroup("Reconstruction Options");
        v.ngui->addVariable("Resolution", resolution);
        v.ngui->addButton("Reset Grid", [&]() {
            // Recreate the grid
            createGrid();
            // Switch view to show the grid
            callback_key_down(v, '3', 0);
        });

        // TODO: Add more parameters to tweak here...
        v.ngui->addVariable("wendlandRadius", wendlandRadius);
        v.ngui->addVariable("polyDegree", polyDegree);
        v.ngui->addVariable("expansionFactor", expansionFactor);

        v.screen->performLayout();
        return false;
    };

    callback_key_down(viewer, '1', 0);

    viewer.launch();
}
