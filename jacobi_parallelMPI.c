#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

double one_jacobi_iteration(double xStart, double yStart,
                          int maxXCount, int maxYCount,
                          double *src, double *dst,
                          double deltaX, double deltaY,
                          double alpha, double omega)
{
#define SRC(XX,YY) src[(YY)*maxXCount+(XX)]
#define DST(XX,YY) dst[(YY)*maxXCount+(XX)]
  int x, y;
  double fX, fY;
  double error = 0.0;
  double updateVal;
  double f;
  // Coefficients
  double cx = 1.0/(deltaX*deltaX);
  double cy = 1.0/(deltaY*deltaY);
  double cc = -2.0*cx-2.0*cy-alpha;

  for (y = 1; y < (maxYCount-1); y++)
  {
      fY = yStart + (y-1)*deltaY;
      for (x = 1; x < (maxXCount-1); x++)
      {
          fX = xStart + (x-1)*deltaX;
          f = -alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY);
          updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
              			(SRC(x,y-1) + SRC(x,y+1))*cy +
              			SRC(x,y)*cc - f
					)/cc;
          DST(x,y) = SRC(x,y) - omega*updateVal;
          error += updateVal*updateVal;
      }
  }
   return error;
}


double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
   return error;
}

int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;
    double maxAcceptableError;
    double local_error;
    double global_error;
    double localError, error;
    double absolute_error;


    double *u = NULL, *u_old = NULL, *tmp = NULL;
    double *u_root = NULL, *u_rootplus = NULL;
    int allocCount;
    int iterationCount, maxIterationCount;
    double t1, t2;
    int my_rank, comm_sz, UP, DOWN, LEFT, RIGHT;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (!((comm_sz==4)||(comm_sz==9)||(comm_sz==16)||(comm_sz==25)||(comm_sz==36)||(comm_sz==49)||(comm_sz==64)||(comm_sz==81))) {
        printf("%s: Doesn't work for number of processes = %d\nTry again!\n", argv[0], comm_sz);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    if (my_rank == 0)
    {
         printf("Input n,m - grid dimension in x,y direction:\n");
         scanf("%d,%d", &n, &m);
         printf("Input alpha - Helmholtz constant:\n");
         scanf("%lf", &alpha);
         printf("Input relax - successive over-relaxation parameter:\n");
         scanf("%lf", &relax);
         printf("Input tol - error tolerance for the iterrative solver:\n");
         scanf("%lf", &tol);
         printf("Input mits - maximum solver iterations:\n");
         scanf("%d", &mits);
         printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, tol, mits);
    }

    MPI_Comm  old_comm, new_comm;
    int ndims, reorder;
    int dim_size[2], periods[2], proccoord[2];
    int coords[2] = {0,0};

    old_comm = MPI_COMM_WORLD;
    ndims = 2;
    dim_size[0] = abs(sqrt(comm_sz));
    dim_size[1] = abs(sqrt(comm_sz));

    periods[0] = 0;
    periods[1] = 0;
    reorder = 1;

    MPI_Cart_create(old_comm, ndims, dim_size, periods, reorder, &new_comm);
    if(my_rank == 0)
    {
      for (int rank=0; rank<comm_sz; rank++)
      {
        MPI_Cart_coords(new_comm, rank, ndims, coords);
      }
    }

    MPI_Cart_coords(new_comm, my_rank, ndims, coords);

    if(coords[0]==0)
        UP = MPI_PROC_NULL;
    else{
        proccoord[0] = coords[0]-1;
        proccoord[1] = coords[1];
        MPI_Cart_rank(new_comm, proccoord, &UP);
    }
    if(coords[0]==abs(sqrt(comm_sz))-1)
        DOWN = MPI_PROC_NULL;
    else{
        proccoord[0] = coords[0]+1;
        proccoord[1] = coords[1];
        MPI_Cart_rank(new_comm, proccoord, &DOWN);
    }
    if(coords[1]==0)
        LEFT = MPI_PROC_NULL;
    else{
        proccoord[0] = coords[0];
        proccoord[1] = coords[1]-1;
        MPI_Cart_rank(new_comm, proccoord, &LEFT);
    }
    if(coords[1]==abs(sqrt(comm_sz))-1)
        RIGHT = MPI_PROC_NULL;
    else{
        proccoord[0] = coords[0];
        proccoord[1] = coords[1]+1;
        MPI_Cart_rank(new_comm, proccoord, &RIGHT);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&relax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mits, 1, MPI_INT, 0, MPI_COMM_WORLD);

    allocCount = (n/sqrt(comm_sz)+2)*(m/sqrt(comm_sz)+2);
    u = 	(double*)calloc(allocCount, sizeof(double));
    u_old = (double*)calloc(allocCount, sizeof(double));
    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory\n");
        exit(1);
    }
    if (my_rank == 0)
    {
        allocCount = (n)*(m);
        u_root = 	(double*)calloc(allocCount, sizeof(double));
        if (u_root == NULL)
        {
            printf("error:  Not enough memory\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
    }

    MPI_Datatype temp_datatype, column_datatype;
    MPI_Type_vector(m/abs(sqrt(comm_sz))+2-2, 1, m/abs(sqrt(comm_sz))+2, MPI_DOUBLE, &temp_datatype);
    MPI_Type_create_resized(temp_datatype, 0, sizeof(double), &column_datatype);
    MPI_Type_commit(&column_datatype);

    MPI_Datatype row_datatype;
    MPI_Type_vector(1, n/abs(sqrt(comm_sz)), m/abs(sqrt(comm_sz)), MPI_DOUBLE, &temp_datatype);
    MPI_Type_create_resized(temp_datatype, 0, sizeof(double), &row_datatype);
    MPI_Type_commit(&row_datatype);

    int sizes[2]    = {n, m};
    int subsizes[2] = {n/abs(sqrt(comm_sz)), m/abs(sqrt(comm_sz))};
    int starts[2]   = {0,0};
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_create_resized(type, 0, (n/abs(sqrt(comm_sz)))*sizeof(double), &subarrtype);
    MPI_Type_commit(&subarrtype);

    int sizesn[2]    = {n/abs(sqrt(comm_sz))+2, m/abs(sqrt(comm_sz))+2};
    int subsizesn[2] = {n/abs(sqrt(comm_sz)), m/abs(sqrt(comm_sz))};
    int startsn[2]   = {1,1};
    MPI_Datatype newtype2;
    MPI_Type_create_subarray(2, sizesn, subsizesn, startsn, MPI_ORDER_C, MPI_DOUBLE, &newtype2);
    MPI_Type_commit(&newtype2);

    int sendcounts[comm_sz];
    int displs[comm_sz];

    if (my_rank == 0)
    {
        for (int i=0; i<comm_sz; i++)
            sendcounts[i] = 1;
        int disp = 0;
        for (int i=0; i<abs(sqrt(comm_sz)); i++) {
            for (int j=0; j<abs(sqrt(comm_sz)); j++) {
                displs[i*abs(sqrt(comm_sz))+j] = disp;
                disp += 1;
            }
            disp += (((n)/abs(sqrt(comm_sz)))-1)*abs(sqrt(comm_sz));

        }
    }
    if (my_rank == 0)
        MPI_Scatterv(u_root , sendcounts, displs, subarrtype, u_old, 1, newtype2,
                     0, MPI_COMM_WORLD);
    else
        MPI_Scatterv(NULL, sendcounts, displs, subarrtype, u_old, 1, newtype2,
                 0, MPI_COMM_WORLD);

    maxIterationCount = mits;
    maxAcceptableError = tol;

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;
    //double deltaX = (xRight-xLeft)/(n-1);
    //double deltaY = (yUp-yBottom)/(m-1);

    double mydeltaX = (xRight-xLeft)/(n/abs(sqrt(comm_sz))-1);
    double mydeltaY = (yUp-yBottom)/(m/abs(sqrt(comm_sz))-1);

    iterationCount = 0;
    local_error = HUGE_VAL;
    global_error = HUGE_VAL;
    clock_t start, diff;

    MPI_Request rreq[4];
    for (int i = 0; i < 4; i++) {
      rreq[i] = MPI_REQUEST_NULL;
    }
    MPI_Request sreq[4];
    for (int i = 0; i < 4; i++) {
      sreq[i] = MPI_REQUEST_NULL;
    }
    MPI_Status stats[8];


    int downhalo = (n/abs(sqrt(comm_sz))+2-1)*(m/abs(sqrt(comm_sz))+2) +1;
    int lefthalo = m/abs(sqrt(comm_sz))+2;
    int righthalo = 2*(m/abs(sqrt(comm_sz))+2)-1;
    int uprow = (m/abs(sqrt(comm_sz))+2)+1;
    int downrow = (n/abs(sqrt(comm_sz))+2-2)*(m/abs(sqrt(comm_sz))+2)+1;
    int leftrow = m/abs(sqrt(comm_sz))+2+1;
    int rightrow = 2*(m/abs(sqrt(comm_sz))+2)-2;
    MPI_Recv_init(u_old +1, 1, row_datatype, UP, 0, MPI_COMM_WORLD, &rreq[2]);            //RECEIVE_TO_UP_HALO
    MPI_Recv_init(u_old + downhalo, 1, row_datatype, DOWN, 0, MPI_COMM_WORLD, &rreq[3]);          //RECEIVE_TO_DOWN_HALO
    MPI_Recv_init(u_old + lefthalo, 1, column_datatype, LEFT, 0,   MPI_COMM_WORLD, &rreq[1]);               //RECEIVE_TO_LEFT_HALO
    MPI_Recv_init(u_old + righthalo, 1, column_datatype, RIGHT, 0, MPI_COMM_WORLD, &rreq[0]);        //RECEIVE_TO_RIGHT_HALO

    MPI_Send_init(u_old + uprow, 1, row_datatype, UP, 0, MPI_COMM_WORLD, &sreq[2]);    //SEND_MY_UP
    MPI_Send_init(u_old + downrow, 1, row_datatype, DOWN, 0, MPI_COMM_WORLD, &sreq[3]);    //SEND_MY_DOWN
    MPI_Send_init(u_old + leftrow, 1, column_datatype, LEFT, 0,   MPI_COMM_WORLD, &sreq[1]);     //SEND_MY_LEFT
    MPI_Send_init(u_old + rightrow, 1, column_datatype, RIGHT, 0, MPI_COMM_WORLD, &sreq[0]);  //SEND_MY_RIGHT

    if(my_rank==0)
        start = clock();
    t1 = MPI_Wtime();

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*WITHOUT CONVERGENCE CHECK*/
    ///*
    if(my_rank==0)
        printf("\nWITHOUT CONVERGENCE CHECK\n");
    while (iterationCount < maxIterationCount && sqrt(local_error)/(n/sqrt(comm_sz)*m/sqrt(comm_sz)) > maxAcceptableError)
    {
        //MPI_Sendrecv(u_old + 2*(m/abs(sqrt(comm_sz))+2)-2, 1, column_datatype, RIGHT, 0,  //SEND_MY_RIGHT
        //             u_old + 2*(m/abs(sqrt(comm_sz))+2)-1 , 1, column_datatype, RIGHT, 0,        //RECEIVE_TO_RIGHT_HALO
        //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Sendrecv(u_old + (m/abs(sqrt(comm_sz))+2)+1 , 1, row_datatype, UP, 0,                                //SEND_MY_UP
        //             u_old +1 , 1, row_datatype, UP, 0,                                       //RECEIVE_TO_UP_HALO
        //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Sendrecv(u_old + m/abs(sqrt(comm_sz))+2+1, 1, column_datatype, LEFT, 0,      //SEND_MY_LEFT
        //             u_old + m/abs(sqrt(comm_sz))+2, 1, column_datatype, LEFT, 0,                //RECEIVE_TO_LEFT_HALO
        //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Sendrecv(u_old + (n/abs(sqrt(comm_sz))+2-2)*(m/abs(sqrt(comm_sz))+2)+1, 1, row_datatype, DOWN, 0,    //SEND_MY_DOWN
        //             u_old + (n/abs(sqrt(comm_sz))+2-1)*(m/abs(sqrt(comm_sz))+2) +1, 1, row_datatype, DOWN, 0,            //RECEIVE_TO_DOWN_HALO
        //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        //MPI_Irecv(u_old +1 , 1, row_datatype, UP, 0, MPI_COMM_WORLD, &rreq[2]);            //RECEIVE_TO_UP_HALO
        //MPI_Irecv(u_old + (n/abs(sqrt(comm_sz))+2-1)*(m/abs(sqrt(comm_sz))+2) +1, 1, row_datatype, DOWN, 0, MPI_COMM_WORLD, &rreq[3]);          //RECEIVE_TO_DOWN_HALO
        //MPI_Irecv(u_old + m/abs(sqrt(comm_sz))+2, 1, column_datatype, LEFT, 0,   MPI_COMM_WORLD, &rreq[1]);               //RECEIVE_TO_LEFT_HALO
        //MPI_Irecv(u_old + 2*(m/abs(sqrt(comm_sz))+2)-1 , 1, column_datatype, RIGHT, 0, MPI_COMM_WORLD, &rreq[0]);        //RECEIVE_TO_RIGHT_HALO
        //MPI_Isend(u_old + (m/abs(sqrt(comm_sz))+2)+1 , 1, row_datatype, UP, 0, MPI_COMM_WORLD, &sreq[2]);    //SEND_MY_UP
        //MPI_Isend(u_old + (n/abs(sqrt(comm_sz))+2-2)*(m/abs(sqrt(comm_sz))+2)+1, 1, row_datatype, DOWN, 0, MPI_COMM_WORLD, &sreq[3]);    //SEND_MY_DOWN
        //MPI_Isend(u_old + m/abs(sqrt(comm_sz))+2+1, 1, column_datatype, LEFT, 0,   MPI_COMM_WORLD, &sreq[1]);     //SEND_MY_LEFT
        //MPI_Isend(u_old + 2*(m/abs(sqrt(comm_sz))+2)-2, 1, column_datatype, RIGHT, 0, MPI_COMM_WORLD, &sreq[0]);  //SEND_MY_RIGHT

        MPI_Startall(4, rreq);
        MPI_Startall(4, sreq);
        MPI_Waitall(4, rreq, stats);

        local_error = one_jacobi_iteration(xLeft, yBottom,
                                     n/sqrt(comm_sz)+2, m/sqrt(comm_sz)+2,
                                     u_old, u,
                                     mydeltaX, mydeltaY,
                                     alpha, relax);

        iterationCount++;
        // Swap the buffers
        MPI_Waitall(4, sreq, stats);

        tmp = u_old;
        u_old = u;
        u = tmp;
    }

    t2 = MPI_Wtime();
    if(my_rank==0)
        diff = clock() - start;
    printf( "rank = %d Iterations=%3d, Elapsed MPI Wall time is %f, Residual is %g\n", my_rank, iterationCount, t2 - t1, sqrt(local_error)/(n/sqrt(comm_sz)*m/sqrt(comm_sz)));
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(my_rank==0){
        printf("\n");
        printf("Statistics: \nNumber of Processes: %d and Size: %d\n",comm_sz,n);
        global_error = sqrt(global_error)/(n*m);
        int msec = diff * 1000 / CLOCKS_PER_SEC;
        printf("Time taken -> %d seconds %d milliseconds\n", msec/1000, msec%1000);
        printf("TIME: %f\n", (double)diff / CLOCKS_PER_SEC);
        printf("Global Residual -> %g\n", global_error);
    }
    //*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*WITH CONVERGENCE CHECK*/
    /*
    if(my_rank==0)
        printf("\nWITH CONVERGENCE CHECK\n");
    while (iterationCount < maxIterationCount && sqrt(global_error)/(n*m) > maxAcceptableError)
    {
        MPI_Startall(4, rreq);
        MPI_Startall(4, sreq);
        MPI_Waitall(4, rreq, stats);

        local_error = one_jacobi_iteration(xLeft, yBottom,
                                     n/abs(sqrt(comm_sz))+2, m/abs(sqrt(comm_sz))+2,
                                     u_old, u,
                                     mydeltaX, mydeltaY,
                                     alpha, relax);
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        iterationCount++;
        MPI_Waitall(4, sreq, stats);

        tmp = u_old;
        u_old = u;
        u = tmp;
    }

    t2 = MPI_Wtime();
    if(my_rank==0)
        diff = clock() - start;
    printf( "rank = %d Iterations=%3d, Elapsed MPI Wall time is %f, Residual is %g\n", my_rank, iterationCount, t2 - t1, abs(sqrt(local_error))/(n/abs(sqrt(comm_sz))*m/abs(sqrt(comm_sz))));
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(my_rank==0){
        printf("\n");
        printf("Statistics: \nNumber of Processes: %d and Size: %d\n",comm_sz,n);
        global_error = sqrt(global_error)/(n*m);
        int msec = diff * 1000 / CLOCKS_PER_SEC;
        printf("Time taken -> %d seconds %d milliseconds\n", msec/1000, msec%1000);
        printf("TIME: %f\n", (double)diff / CLOCKS_PER_SEC);
        printf("Global Residual -> %g\n", global_error);
    }
    */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    /*
    if (my_rank == 0)
        MPI_Gatherv( u_old, 1, newtype2 ,
                     u_root, sendcounts , displs , subarrtype ,
                     0, MPI_COMM_WORLD);
    else
         MPI_Gatherv( u_old, 1, newtype2 ,
                 NULL , sendcounts , displs , subarrtype ,
                 0, MPI_COMM_WORLD);

    if (my_rank==0)
    {
        allocCount = (n+2)*(m+2);
        u_rootplus = 	(double*)calloc(allocCount, sizeof(double));
        if (u_rootplus == NULL)
        {
            printf("error:  Not enough memory\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
        MPI_Datatype plus;
        MPI_Type_vector(n, m, m+2, MPI_DOUBLE, &plus);
        MPI_Type_commit(&plus);

        MPI_Sendrecv(u_root ,n*m , MPI_DOUBLE, 0, 0,
                     u_rootplus + n+3, 1, plus, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        absolute_error = checkSolution(xLeft, yBottom,
                                      n+2, m+2,
                                      u_rootplus,
                                      deltaX, deltaY,
                                      alpha);
        absolute_error = sqrt(absolute_error)/(n*m);
        printf("Error of the iterative solution -> %g\n", absolute_error);
        free(u_rootplus);
    }
    */
    local_error = checkSolution(xLeft, yBottom,
                               n/abs(sqrt(comm_sz))+2, m/abs(sqrt(comm_sz))+2,
                               u_old,
                               mydeltaX, mydeltaY,
                               alpha);
    MPI_Allreduce(&local_error, &absolute_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (my_rank == 0){
        absolute_error = sqrt(absolute_error)/(n*m);
        printf("Error of the iterative solution -> %g\n", absolute_error);
    }
    MPI_Type_free(&subarrtype);
    MPI_Type_free(&newtype2);
    MPI_Type_free(&row_datatype);
    MPI_Type_free(&column_datatype);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    free(u);
    free(u_old);
    if (my_rank == 0)
        free(u_root);

    return 0;
}
