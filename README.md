
##### TL;DR
This project was linked to a lecture about Shape Modeling. This project presents a way to compute the surface of a point cloud, with normals.
Note that without normals, we would need preprocess the point cloud by creating local plan, to produce normals ; and by using a spanning tree, to propagate one global direction (ext/int) to all normals.
<br/>
The goal to find a function, which has a 0-value on the surface. It has a positive value outside the mesh, a negative value inside the mesh.
<br/>
This problem is similar to an interpolation problem, and therefore we need to create a constraint system to solve (the value the function should have at each known point).
<br/>

##### Table of Contents  
[1. Setting up the Constraints](#constraints)  <br/>
[1.1. Creating the constraints](#creating)  <br/>
[1.2. Implementing a spatial index to accelerate neighbour calculations](#index)  <br/>
[2. MLS Interpolation](#MLS)  <br/>
[2.1. Create a grid sampling the 3D space](#grid)  <br/>
[2.2. Evaluate the implicit function on the grid](#implicit)  <br/>
[2.3. Using a non-axis-aligned grid](#non-axis-aligned)  <br/>
[3. Extracting the surface](#extracting)  <br/>

<a name="constraints"/>

#### 1. Setting up the Constraints

<a name="creating"/>

#### 1.1. Creating the constraints

We create constraints to interpolate the surface function.
If we only use constraint with the format f(x)=0 at x, with x are surface point, the trivial solution found will be the null-function (f(x)=0 for all x)
<br/>
So we have to create points with values different from 0.
For each surface point of the cloud point, we create 2 new points : a point "inside" the mesh, with a negative value (-Epsilon) ; and a point "outside" the mesh, with a positive value. (+Epsilon)
<br/>
We plot each constraint point in a color chosen based on its type (inside/outside/surface)
<br/>

###### Usage

We calculate the distance at which the two new points will be distant from the original surface point. We have to be sure, when we establish the two new points, that it does not intersect with new neighboor of other original points.
A property has to be respected : the two created neighboors has to be the two closer point of the original surface point.
<br/>
If not, we reduce the distance at wich the point will be distant from the original surface point. We reduce this distance iteratively, while the two neighboor does not satisfy this property.
<br/>
We have to be aware to start with a global epsilon value (determined as a fraction of the diagonal of the bounded box of the mesh), that we decrease each time the nearest neighboor is closer than the tested value.
<br/>

```C++
            double localEspPlus = globalEps;

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

            //Add the point in the data structure
            constrained_points.row(i) << P.row(i);
            constrained_points.row(i + N.rows()) << pointEspPlus.row(0);
            constrained_points.row(i + 2 * N.rows()) << pointEspMinus.row(0);

            //Add the constraint value for each point
            constrained_values.row(i)[0] = 0;
            constrained_values.row(i + N.rows())[0] = localEspPlus;
            constrained_values.row(i + 2 * N.rows())[0] = localEspMinus;
        }
(...)
        //Last class set of points = interieur points = -Epsilon
        viewer.data.add_points(constrained_points.bottomRows(N.rows()), Eigen::RowVector3d(255,255,0));
 
```

###### Result
<div style="text-align:center">
<p align="center">
  <img style="height : 300px;" src="/results/sphereTriple.gif">
<br/>RED = +Epsilon point; BLACK = 0 point; YELLOW = -Epsilon<br/>  
  <img style="height : 300px;" src="/results/bunny1000Triple.gif">
  <img style="height : 300px;" src="/results/bunny500Triple.gif">
  <br/>Left = bunny-1000 ; Right = bunny-500<br/>  
  <img style="height : 300px;" src="/results/catTriple.gif">
  <img style="height : 300px;" src="/results/houndTriple.gif">
  <img style="height : 300px;" src="/results/horseTriple.gif">
  <img style="height : 300px;" src="/results/luigiTriple.gif">
</p>
</div>

__________________________________

<a name="index"/>

#### 1.2. Implementing a spatial index to accelerate neighbour calculations

The previous implementation is computation-consuming. (at least O(n²))
<br/>
We implement a data structure (imposed by the lecture) that will allow : 
* to request the the closest original point from a certain position in space.
* to request all original points within distance h from a certain position in space (needed to select constraints with nonzero weight).

###### Usage
The data structure implemented is basically a list of linked list. (LinkedListCell)
Each entry of the first list is a called a cell, and each linked-list binded to a cell is composed by the points in this cell.
<br/>
The size of a cell is defined by the parameter "wendland radius". 
This parameter give the radius in which neighboor can have an influence on the calculation of the position of the surface.
Each cell has a size of (wendland radius)*(wendland radius)*(wendland radius). 
<br/>
<p align="center">
  <img style="height : 550px;" src="/results/schema.jpg">
</p>
<br/>
That way, we only need - and we are sure it's sufficient - to visit the 27 directs neighboors (9 on top, 9 below, 8 around) of the cell we consider, to get get through all the potential neighboors of the current point considered, in the wendland radius.
<br/>
Some tricks are used, for example : 
* We need to have more cells than [(the length of the bounding box of the mesh)/(wendland radius)] because this value is not round. So, extra cells has to be created at the edges of the grid. (See previous drawing, the green box, to have a visual representation of it)
* The 3D grid of cells created is store as a 1D list. Therefore, we need to convert 3D coordinates to 1D coordinates. Functions has been created to achieve this task.
<br/>

```C++
    //Conversion Coords 3D of a Cell ==> Coords 1D of a cell
    unsigned int getIndiceFromCoord(unsigned int x, unsigned int y, unsigned int z)
    {
        return (x * nbCellXEdge + y) * nbCellYEdge + z;
    }

    //Conversion Coords 3D of a point ==> Coords 1D of the cell it is in.
    unsigned int getIndiceFromConstraintsCoord(int xConstraintCoord, int yConstraintCoord, int zConstraintCoord)
    {
        (...) //Convert 3D coords in 1D coords
    }

    //Coords 1D of a cell ==> List of the 1D coords of the neighbooring cells.
    std::vector<int> getAdjacentsCellsFromIndice(unsigned int index)
    {
        (...) // Convert 1D coords in 3D coords.
        return getAdjacentsCells(x, y, z);
    }

    //Compute the 3D coordinates of the neighboors cells of a given cell, call a conversion to 1D coordinate, store it.
    std::vector<int> getAdjacentsCells(unsigned int x, unsigned int y, unsigned int z)
    {
        (...)
        for (int i = -1; i < 2; i++)
                for (int j = -1; j < 2; j++)
                        for (int k = -1; k < 2; k++)
                                listAdjacentsCells.push_back(getIndiceFromCoord(x + i, y + j, z + k));

    }

    //Initialize data structure. Necessary called after a modification (rotation, translation,etc.) of the points of the mesh. It populates the quick access data structure
    void prepareQuickAccess()
    {
        (...)
        // Grid spacing = we want cells of Wendland radius size
        dx = wendlandRadius; //(double)(resolution - 1);
        dy = wendlandRadius; //(double)(resolution - 1);
        dz = wendlandRadius; //(double)(resolution - 1);

        (...)
        //We are creating +1 cell because the wendland cells are not aligned with the grid
        nbCellXEdge = ((dim[0] + extraSpace[0]) / wendlandRadius) + 1;
        nbCellYEdge = ((dim[1] + extraSpace[1]) / wendlandRadius) + 1;
        nbCellZEdge = ((dim[2] + extraSpace[2]) / wendlandRadius) + 1;

        (...)
        //Clean the structure
        (...)

        //For all points, calculate in which cell it should belongs to. Store it.
        unsigned int coordCell = 0;
        for (int i = 0; i < constrained_points.rows(); i++)
        {
            coordCell = getIndiceFromConstraintsCoord(constrained_points.row(i)[0], constrained_points.row(i)[1], constrained_points.row(i)[2]);
            quikStructure[coordCell].push_back(i);
        }
    }

    // Give the closest point of the point "pointQ" using the quick data structure
    int getClosest(int pointQ)
    {
        (...)
        //Get the cell in which the P dot is.
        unsigned int coordCell = getIndiceFromConstraintsCoord(refCoords.row(0)[0], refCoords.row(0)[1], refCoords.row(0)[2]);

        //Is there points in the cell of the reference Point ?
        if (quikStructure[coordCell].size() != 0)
        {
            //Cas 1 : Neighboors in his own cell
            for (int i = 0; i < quikStructure[coordCell].size(); i++)
            {
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
        (...) // Other cases : look in the neighbooring cells, etc.

        return nearestNeighboor;
    }

    // Give the closest set of point of the point "pointQ" using the quick data structure, in the specified radius
    Eigen::Matrix<int, Eigen::Dynamic, 1> getClosestSetConstrained(int indicePointGrid, double rayon)
    {
        (...)
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
                (...)

                //Compare
                if (rayon > curDistanceVal)
                {
                    //We found a good value, we add it to the list
                    closestPointsList.row(nbNeighboor) << quikStructure[cellsToVisit[c]][i];
                    nbNeighboor++;
                }
            }
        }
        (...)
    }
```

###### Result

**An impressive gain in time is noticed.** No exacts numbers are available (more developement would be necessary to precisely mesure the time of the gain, with and without the new quick data structure.)
<br/>
However, we can rougly determine the new data structure improve by a factor 100 to 500 the speed of computation. (21 sec to 0.05 sec, for example. See below examples for rough values)
<br/>
**Note that, the wendland radius is directly correlated to performances. A bad wendland radius leads to bad performances. This choice is crucial, but can be automated !**

__________________________________


<a name="MLS"/>

#### 2. MLS Interpolation

<a name="grid"/>

#### 2.1. Create a grid sampling the 3D space

After the creation of the constraints, we want to construct the implicit function.
To get this implicit function, we can evaluate this function at each point of a new grid, covering the whole space.
<br/>
The grid resolution is configured by the global variable resolution, which can be changed, axis-aligned with the bounding box of the point cloud, and slightly enlarged.
<br/>


__________________________________


<a name="implicit"/>

#### 2.2. Evaluate the implicit function on the grid

We evaluate the implicit function for each grid node. 
We use a moving least squares approximation, with the wendland radius as parameter.

###### Documentation informations
colPivHouseholderQr() has been used from Eigen library, to solve the system that we end with.
Informations about the function : [-> Documentation](https://eigen.tuxfamily.org/dox/classEigen_1_1ColPivHouseholderQR.html)

###### Usage
For each grid point, we evaluate the value of the implicit function.
We calculate the values of the coefficients of the function for the specific current grid point. For this, we solve a system, with the previously constructed constraints.

The time of reconstruction had been measured.
**For example for the sphere :**
- Without the Data structure : 21 sec
- With the Data structure : 0.05 sec
(Resolution = 10 / WendlandR = 5 / PolyDegree = 1) 

**For example for the cat :**
- Without the Data structure : 210 sec
- With the Data structure : 0.16 sec
(Resolution = 30 / WendlandR = 100 / PolyDegree = 1) 

Note an approximation : if we use the data structure with a too large wendland radius the cells are not used to improve performance. This is only an approximation of the bare method, even if it's not correct due to the system resolution that is way bigger.

```C++
    void evaluateImplicitFunc()
    {
        time.start();
        (...Initialisation...)

        //Iter through all gridpoints
        for (unsigned int x = 0; x < resolution; ++x){
            for (unsigned int y = 0; y < resolution; ++y){
                for (unsigned int z = 0; z < resolution; ++z){

                    //Get the actual coordinates of the grid point
                    double xCoordGrid = grid_points.row(currentIndex)[0];
                    double yCoordGrid = grid_points.row(currentIndex)[1];
                    double zCoordGrid = grid_points.row(currentIndex)[2];

                    (...)
                    //We get his neighboorhood (in the wenderradius)
                    Eigen::Matrix<int, Eigen::Dynamic, 1> resultSet = getClosestSetConstrained(currentIndex, wendlandRadius);

                    (...)

                    if (resultSet.rows() >= nbBases)
                    { //The current point of the grid has some neighboors

                        //For each point of the neighboorhood :
                        for (int i = 0; i < resultSet.rows(); i++)
                        {
                            //We get the id of this points (ID valid in contrainted_points, constrained_value, etc.)
                            long idNeighboorPoint = resultSet.row(i)[0];

                            //We get the constraint of this point & We store the constraint
                            double constValue = constrained_values.row(idNeighboorPoint)[0];

                            //We get the coordinates of this point
                            currentCoords.row(0) << constrained_points.row(idNeighboorPoint);

                            //Calcul distance between actualPoint and neighboor
                            Eigen::RowVector3d curDistance = grid_points.row(currentIndex) - currentCoords.row(0);
                            double curDistanceVal = sqrt(curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2]);

                            //We calcul the wendeland weight factor for this point = Weight of this neighboor in the equation
                            double wendlandValue = pow((1 - (curDistanceVal / wendlandRadius)), 4) * ((4 * curDistanceVal / wendlandRadius) + 1);

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

                        //We solve the system (Eigen call)
                        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(resultSet.rows(), nbBases);
                        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(resultSet.rows(), 1);
                        //Eigen::VectorXd coefs = Eigen::VectorXd::Zero(nbBases,1);

                        A = weights * base;
                        B = weights * constraintsVal;

                        Eigen::VectorXd coefs = A.colPivHouseholderQr().solve(B);

                        //We compute the value of the function for this particular point
                        double functionValue = 0;
                        for (int j = 0; j < nbBases; j++)
                        {
                            functionValue += coefs[j] * baseGrid.row(0)[j]; // We do coef[0]*1 + coef[1]*x + coef[2]*y + ...
                        }

                        //We store the value
                        grid_values[currentIndex] = functionValue;
                    }
                    else
                    {
                        //The current point of the grid has NO neighboors
                        grid_values[currentIndex] = 100;
                        //We store an arbitrary high value
                    }
                }
            }
        }
        time.stop();
    }
```

###### Result
<div style="text-align:center">
<p align="center">
  <img style="height : 300px;" src="/results/bunny1000Grid.gif">
  <img style="height : 300px;" src="/results/bunny_grid_R40_W0.5_P2.png">
  <br/>Left : Mesh = bunny-1000 / Resolution = 40 / WendlandR = 0.5 / PolyDegree = 2 
  <br/>Right : Mesh = bunny-1000 / idem <br/>  
   
  <img style="height : 300px;" src="/results/bunny500Grid.gif">
    <img style="height : 300px;" src="/results/bunny_grid_R60_W0.05_P2.png">
      <br/>Left : Mesh = bunny-500 / Resolution = 30 / WendlandR = 100 / PolyDegree = 1 
  <br/>Right : Mesh = bunny-500 / Resolution = 60 / WendlandR = 0.05 / PolyDegree = 2 / Sampling time : 36.8013 sec<br/>  
  
  <img style="height : 300px;" src="/results/catGrid.gif">
  <img style="height : 300px;" src="/results/cat_grid_R100_W80_P2.png">
  <br/>Left : Mesh = cat / Resolution = 30 / WendlandR = 100 / PolyDegree = 1 
  <br/>Right : Mesh = cat / Resolution = 100 / WendlandR = 80 / PolyDegree = 2 <br/>  

  <img style="height : 300px;" src="/results/houndGrid.gif">
    <img style="height : 300px;" src="/results/hound_R100_W0.01_P2.png">
    <br/>Left : Mesh = hound / Resolution = 25 / WendlandR = 0.05 / PolyDegree = 0 / Sampling time : 730.426 sec
    <br/>Right : Mesh = hound / Resolution = 100 / secWendlandR = 0.01 / PolyDegree = 2 <br/>  
    
  <img style="height : 300px;" src="/results/horseGrid.gif">
    <img style="height : 300px;" src="/results/horse_grid_R50_W0.03_P0.png">
      <br/>Left : Mesh = horse / Resolution = 40 / WendlandR = 0.01 / PolyDegree = 0 / Sampling time : 61.0157 sec
            <br/>Right : Mesh = horse / Resolution = 50 / WendlandR = 0.03 / PolyDegree = 0 / Sampling time : 134.786 sec<br/>  

  <img style="height : 300px;" src="/results/luigiGrid.gif">
      <img style="height : 300px;" src="/results/luigi_grid_R40_W10_P1.png">
      <br/>Left : Mesh = luigi / Resolution = 25 / WendlandR = 5 / PolyDegree = 0 
      <br/>Right : Mesh = luigi / Resolution = 40 / WendlandR = 10 / PolyDegree = 1 / Sampling time : 2.38913 sec<br/>  
  
  <img style="height : 300px;" src="/results/sphereGrid.gif">
      <img style="height : 300px;" src="/results/sphere_R20_W1_P2.png">
        <br/>Left : Mesh = sphere / Resolution = 10 / WendlandR = 5 / PolyDegree = 1
            <br/>Right : Mesh = sphere / Resolution = 20 / WendlandR = 1 / PolyDegree = 2 <br/>  
  </p>
</div>

__________________________________

<a name="non-axis-aligned"/>

#### 2.3. Using a non-axis-aligned grid

Point cloud may be not aligned with grid axis. Therefore, huge part of the grid on which we evaluate the implicit function are "empty" and so generate irrelevant calculations.
<br/>
We add a previous calculation to align the cloud point with the grid-axis, based on PCA. (Principale Components Analysis)
<br/>

###### Documentation informations
Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>  had been used, mainly to get the PCA extraced Eigen vectors.
Informations about the function : [-> Documentation](https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html)

Eigen::Quaternionf() had been used, mainly to establish the rotation between PCA vector and initial basis.
Informations about the function : [-> Documentation](https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html)

###### Usage
The main idea is to **compute a PCA** on the cloud points of the initial mesh. 
<br/>
We get 3 vectors correspondings (depending on the order of the eigen values) to the 3 mains axes of the mesh.
Then, the goal is to **compute a rotation matrix** between the "main" axe of the mesh and one of the three basis vector (<1,0,0> for example). 
Then we have to **apply this transformation** to all the points of P (initial coordinates of the points), constrained_points (coordinates of the initials points and their +Eps/-Eps neighboors) and N (the normals of the initials points).
<br/>
After the rotation, during the grid creation, it will **automatically fits** as closely as possible to the new aligned mesh.
<br/>
One function call had been added in the "createGrid()" function, to optimize the orientation of the mesh as explained..
<br/>

```C++
void createGrid()
    {
        <Already implemented>
        //Only one line added. See the alignement of axis
        optimizePosition();
    }

    (...)

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

        // ## Rotation preparation :
        Eigen::Matrix3f R = Eigen::Quaternionf().setFromTwoVectors(eigenC, target).toRotationMatrix().cast<float>();

        // ## Rotation computation :
        Eigen::MatrixXd constrained_pointsTMP = Eigen::MatrixXd::Zero(constrained_points.rows(), 3);
        Eigen::MatrixXd PTMP = Eigen::MatrixXd::Zero(P.rows(), 3);
        Eigen::MatrixXd NTMP = Eigen::MatrixXd::Zero(N.rows(), 3);

        //We calculate the rotation matrix for the normals.
        Eigen::Matrix3d normalMatrix = R.inverse().transpose().cast<double>();

        //We rotate the normals
        for (int i = 0; i < N.rows(); i++)
        {
            Eigen::Vector3d norTMP;
            Eigen::Vector3d norOUT;

            norTMP = N.row(i).transpose();

            //norOUT = R.cast<double>() * norTMP;
            norOUT = (normalMatrix * norTMP); // .normalize()

            NTMP.row(i)[0] = norOUT[0];
            NTMP.row(i)[1] = norOUT[1];
            NTMP.row(i)[2] = norOUT[2];
        }

        //We rotate the constrained points
        for (int i = 0; i < constrained_points.rows(); i++)
        {
            Eigen::Vector3d vecTMP;
            Eigen::Vector3d vecOUT;

            vecTMP = constrained_points.row(i).transpose();

            vecOUT = R.cast<double>() * vecTMP;

            constrained_pointsTMP.row(i)[0] = vecOUT[0];
            constrained_pointsTMP.row(i)[1] = vecOUT[1];
            constrained_pointsTMP.row(i)[2] = vecOUT[2];
        }

        //We rotate the initials points
        for (int i = 0; i < P.rows(); i++)
        {
            Eigen::Vector3d vecTMPP;
            Eigen::Vector3d vecOUTP;

            vecTMPP = P.row(i).transpose();

            vecOUTP = R.cast<double>() * vecTMPP;

            PTMP.row(i)[0] = vecOUTP[0];
            PTMP.row(i)[1] = vecOUTP[1];
            PTMP.row(i)[2] = vecOUTP[2];
        }

        //We save the rotated entities
        constrained_points = constrained_pointsTMP;
        P = PTMP;
        N = NTMP;

        //Recalculate necessary values
        (...)
```

###### Result
<div style="text-align:center">
<p align="center">
    <img style="height : 300px;" src="/results/luigiBeforePCAR40W10P1.gif">
    <img style="height : 300px;" src="/results/luigiBeforePCA_R40_W10_P1.png">
    <br/>Left : Mesh = luigi without PCA / Resolution = 40 / WendlandR = 10 / PolyDegree = 1 
    <br/>Right : Mesh = luigi without PCA / Resolution = 40 / WendlandR = 10 / polyDegree = 1 / Sampling time : 1.94553 sec <br/>
    <img style="height : 300px;" src="/results/luigiAfterPCAR40W10P1.gif">
    <img style="height : 300px;" src="/results/luigiAfterPCA_R40_W10_P1.png">
    <br/>Left : Mesh = luigi without PCA / Resolution = 40 / WendlandR = 10 / PolyDegree = 1 
    <br/>Right : Mesh = luigi without PCA / Resolution = 40 / WendlandR = 10 / PolyDegree = 1 / Sampling time : 2.4162 sec<br/>  
</p>
</div>

__________________________________

<a name="extracting"/>

### 2.3. Extracting the surface

We finnaly extract the surface from the valued grid generated.
We use a marching cubes algorithm, to extract the zero isosurface from the grid.
<br/>
A press on key ’4’ generate the surface and display it.

###### Usage
Briefly, we can see the parameters greatly influence the final result.

We could do some improvements to the previous algorithms, as an **automatic selection of the wendland radius value**. It's would be quite easy to implement, but would need some iterated process to raffine to a good value.
To select the wendland radius, we could just iterate (dichotomy) on an hardcoded range of values, and we could select a "good" one, thanks to the number of cells of the quick data structure. There is tow "bad" cases : 
- If **the number of cell is too low**, the process will be very slow (we will need to iterate over all points), and so, the wendland radius is too big (we will considerate more than the required number of neighboor in our solving of the system)
- If **the number of cell is too high** (higher than the number of point), the process will be fast but inefficient, because the wendland radius will be too small (we won't find any neighboor for most of the points, and so the mesh won't be reconstructed)

Then, we can target (as a first approximation) a wendland radius to satisfy this equality : 
**(number Of Points in the mesh)/(Nb Terme In The Selected degree) = (number of cells in the quick data structure)**

For example, in degree 1, if we have a 1000 points's mesh, we would try to find the wendland radius to have 1000/4 = 250 cells in the data structure. This way, we globally have enough point per cells to solve systems, but not too much, to keep good performances.

Concerning the polynomial degree, **the smoother is the lower degree** (0). It seems the higher the polynomial degree is, the less numerically stable it is. The "smoothness" is also greatly influence my the wendland radius when the polynomial degree is high.

As a first approximation, we could consider a **subdivision process** in order to "smooth" the final mesh.

###### Result
<div style="text-align:center">
<p align="center">
  <img style="height : 300px;" src="/results/bunny1000Surface.gif">
  <img style="height : 300px;" src="/results/bunny500Surface.gif">
  <img style="height : 300px;" src="/results/catSurface.gif">
  <img style="height : 300px;" src="/results/houndGood.gif">
  <img style="height : 300px;" src="/results/horseGood.gif">
  <img style="height : 300px;" src="/results/luigiSurface.gif">
  <img style="height : 300px;" src="/results/sphereSurface.gif"></p>
</div>

Note that the code is not optimized by thoughtful laziness: this code is not made to be reused as is in other applications, and if this is the case, the code will be reviewed and optimized. Thanks for your understanding.
