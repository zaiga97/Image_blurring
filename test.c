#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void halo_exc(unsigned short *b, int k_s, int blockrows, int blockcols, int sblockrows, int sblockcols, MPI_Comm GRID_COMM_WORLD);

int xsize = 10;
int ysize = 6;
unsigned short int *a;

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int numprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
    	a = (unsigned short int*)malloc(xsize * ysize * sizeof(short int));
    	for (int i = 0; i < xsize * ysize; ++i)
    	{
    		a[i] = i;
    	}
    }

    int k = 2;
    int dim[2], period[2];
    int coords[2];
	int reorder = 1;
	dim[0] = 0;
	dim[1] = 0;
	period[0] = 0;
	period[1] = 0;
    MPI_Dims_create(numprocs, 2, dim);
  	MPI_Comm GRID_COMM_WORLD;
  	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &GRID_COMM_WORLD);

  	MPI_Comm_size(GRID_COMM_WORLD, &numprocs);
  	MPI_Comm_rank(GRID_COMM_WORLD, &rank);

  	MPI_Cart_coords(GRID_COMM_WORLD, rank, 2, coords);

  	printf("dim: (%d,%d)\n", dim[0],dim[1]);

  	int npcol = dim[0];
  	int nprow = dim[1];
  	int blockrows = ysize/nprow;
  	int blockcols = xsize/npcol;

  	int sblockrows = ysize/nprow + 2*k;
  	int sblockcols = xsize/npcol + 2*k;

  	unsigned short b[sblockrows * sblockcols];

  	MPI_Datatype blocktype;
  	MPI_Datatype blocktype2;
    MPI_Type_vector(blockrows, blockcols, xsize, MPI_UNSIGNED_SHORT, &blocktype2);
    MPI_Type_create_resized( blocktype2, 0, sizeof(unsigned short int), &blocktype);
    MPI_Type_commit(&blocktype);

    MPI_Datatype sblocktype;
  	MPI_Datatype sblocktype2;
    MPI_Type_vector(blockrows, blockcols, sblockcols, MPI_UNSIGNED_SHORT, &sblocktype2);
    MPI_Type_create_resized( sblocktype2, 0, sizeof(unsigned short int), &sblocktype);
    MPI_Type_commit(&sblocktype);


    int disps[nprow*npcol];
    int counts[nprow*npcol];
    for (int ii=0; ii<nprow; ii++) {
        for (int jj=0; jj<npcol; jj++) {
            disps[ii*npcol+jj] = ii*xsize*blockrows+jj*blockcols;
            counts [ii*npcol+jj] = 1;
        }
    }


    MPI_Barrier(GRID_COMM_WORLD);

    MPI_Scatterv(a, counts, disps, blocktype, &b[k * sblockcols + k], blockrows*blockcols, sblocktype, 0, GRID_COMM_WORLD);

    /* each proc prints it's "b" out, in order */
    for (int proc=0; proc < numprocs; proc++) {
        if (proc == rank) {
            printf("Rank = %d\n", rank);
            if (rank == 0) {
                printf("Global matrix: \n");
                for (int ii=0; ii < ysize; ii++) {
                    for (int jj=0; jj < xsize; jj++) {
                        printf("%3d ",(int)a[ii*xsize+jj]);
                    }
                    printf("\n");
                }
            }
            printf("Local Matrix:\n");
            for (int ii=0; ii < sblockrows; ii++) {
                for (int jj=0; jj< sblockcols; jj++) {
                    printf("%3d ",b[ii*sblockcols+jj]);
                }
                printf("\n");
            }
            printf("\n");
        }
        MPI_Barrier(GRID_COMM_WORLD);
    }

    halo_exc(&b, k, blockrows, blockcols, sblockrows, sblockcols, GRID_COMM_WORLD);

    //blur con openmp

    MPI_Gatherv(&b[k * sblockcols + k], 1, sblocktype, a, counts, disps, blocktype, 0, GRID_COMM_WORLD);
   	MPI_Barrier(GRID_COMM_WORLD);

   	for (int proc=0; proc < numprocs; proc++) {
        if (proc == rank) {
            printf("Rank = %d\n", rank);
            if (rank == 0) {
                printf("Global matrix: \n");
                for (int ii=0; ii < ysize; ii++) {
                    for (int jj=0; jj < xsize; jj++) {
                        printf("%3d ",(int)a[ii*xsize+jj]);
                    }
                    printf("\n");
                }
            }
            printf("Local Matrix:\n");
            for (int ii=0; ii < sblockrows; ii++) {
                for (int jj=0; jj< sblockcols; jj++) {
                    printf("%3d ",b[ii*sblockcols+jj]);
                }
                printf("\n");
            }
            printf("\n");
        }
        MPI_Barrier(GRID_COMM_WORLD);
    }



    MPI_Finalize();

    return 0;
}


void halo_exc(unsigned short *b, int k_s, int blockrows, int blockcols, int sblockrows, int sblockcols, MPI_Comm GRID_COMM_WORLD)
{
	int source;
  	int dest;
  	MPI_Request request;
  	MPI_Status status;

    MPI_Datatype vsblocktype;
  	MPI_Datatype vsblocktype2;
    MPI_Type_vector(blockrows, k_s, sblockcols, MPI_UNSIGNED_SHORT, &vsblocktype2);
    MPI_Type_create_resized( vsblocktype2, 0, sizeof(unsigned short int), &vsblocktype);
    MPI_Type_commit(&vsblocktype);

    MPI_Datatype osblocktype;
  	MPI_Datatype osblocktype2;
    MPI_Type_vector(k_s, sblockcols, sblockcols, MPI_UNSIGNED_SHORT, &osblocktype2);
    MPI_Type_create_resized( osblocktype2, 0, sizeof(unsigned short int), &osblocktype);
    MPI_Type_commit(&osblocktype);

// x + 1
    MPI_Cart_shift(GRID_COMM_WORLD, 0, 1, &source, &dest);
    if (source != MPI_PROC_NULL)
    {
    	MPI_Irecv(&b[k_s*sblockcols], 1, vsblocktype, source, 1, GRID_COMM_WORLD, &request);
    }
    else
    {
    	for (int i = k_s; i < blockrows + k_s; ++i)
    	{
    		for (int j = 0; j < k_s; ++j)
    		{
    			b[i*sblockcols + j] = 0;
    		}
    	}
    }


    if (dest != MPI_PROC_NULL)
    {
    	MPI_Send(&b[k_s*sblockcols + blockcols], 1, vsblocktype, dest, 1, GRID_COMM_WORLD);
    }

    if (source != MPI_PROC_NULL)
    {
    	MPI_Wait(&request, &status);
    }

// x - 1
    MPI_Cart_shift(GRID_COMM_WORLD, 0, -1, &source, &dest);
    if (source != MPI_PROC_NULL)
    {
    	MPI_Irecv(&b[k_s*sblockcols + k_s +blockcols], 1, vsblocktype, source, 2, GRID_COMM_WORLD, &request);
    }
    else
    {
    	for (int i = k_s; i < blockrows + k_s; ++i)
    	{
    		for (int j = k_s + blockcols; j < 2*k_s + blockcols; ++j)
    		{
    			b[i*sblockcols + j] = 0;
    		}
    	}
    }


    if (dest != MPI_PROC_NULL)
    {
    	MPI_Send(&b[k_s*sblockcols + k_s], 1, vsblocktype, dest, 2, GRID_COMM_WORLD);
    }

    if (source != MPI_PROC_NULL)
    {
    	MPI_Wait(&request, &status);
    }

// y + 1
    MPI_Cart_shift(GRID_COMM_WORLD, 1, 1, &source, &dest);
    if (source != MPI_PROC_NULL)
    {
    	MPI_Irecv(b, 1, osblocktype, source, 3, GRID_COMM_WORLD, &request);
    }
    else
    {
    	for (int i = 0; i < k_s; ++i)
    	{
    		for (int j = 0; j < 2*k_s + blockcols; ++j)
    		{
    			b[i*sblockcols + j] = 0;
    		}
    	}
    }


    if (dest != MPI_PROC_NULL)
    {
    	MPI_Send(&b[blockrows*sblockcols], 1, osblocktype, dest, 3, GRID_COMM_WORLD);
    }

    if (source != MPI_PROC_NULL)
    {
    	MPI_Wait(&request, &status);
    }

// y - 1
    MPI_Cart_shift(GRID_COMM_WORLD, 1, -1, &source, &dest);
    if (source != MPI_PROC_NULL)
    {
    	MPI_Irecv(&b[(blockrows + k_s)*sblockcols], 1, osblocktype, source, 4, GRID_COMM_WORLD, &request);
    }
    else
    {
    	for (int i = (k_s+blockrows); i < 2*k_s+blockrows; ++i)
    	{
    		for (int j = 0; j < 2*k_s + blockcols; ++j)
    		{
    			b[i*sblockcols + j] = 0;
    		}
    	}
    }


    if (dest != MPI_PROC_NULL)
    {
    	MPI_Send(&b[k_s*sblockcols], 1, osblocktype, dest, 4, GRID_COMM_WORLD);
    }

    if (source != MPI_PROC_NULL)
    {
    	MPI_Wait(&request, &status);
    }
}
