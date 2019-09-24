
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int startrow, const int endrow, const int nx, const int ny, float **restrict image, float **restrict tmp_image);
int calcStart(int nx, int rank, int size);
int calcEnd(int nx, int rank, int size);
void init_image(const int nx, const int ny, float **  image, float **  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float **  image);
double wtime(void);

int main(int argc, char *argv[]) {

  int myrank;
  int size;
  int startrow;
  int endrow;
  MPI_Status status;
  MPI_Request request, startrequest, endrequest;

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // MPI_Init returns once it has started up processes
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &myrank );

  if (myrank == MASTER) {
    printf("NUMBER OF PROCESSORS = %d\nBOARD SIZE = %d x %d\n", size, nx, ny);
  }


  // Allocate the 2D image arrays
  float **restrict image = (float **) malloc(sizeof(float *)*nx);
  for (int i = 0; i < nx; i++) image[i] = (float *) malloc(ny*sizeof(float));
  float **restrict tmp_image = (float **) malloc(sizeof(float *)*nx);
  for (int i = 0; i < nx; i++) tmp_image[i] = (float *) malloc(ny*sizeof(float));

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // Calculate the start and end row of the current process
  startrow = calcStart(nx, myrank, size);
  endrow = calcEnd(nx, myrank, size);

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    // Run stencil on processes' section of board
    stencil(startrow, endrow, nx, ny, image, tmp_image);
    // Send/receive first and last rows to neighboring processes.
    if (size > 1) {
      if (myrank == MASTER) {
        MPI_Isend(tmp_image[endrow-1], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &request);
      }
      else if (myrank == size-1) {
        MPI_Irecv(tmp_image[startrow-1], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &startrequest);
      }
      else if (myrank%2 == 0) {
        MPI_Isend(tmp_image[endrow-1], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &request);
        MPI_Irecv(tmp_image[startrow-1], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &startrequest);
      }
      else if (myrank%2 == 1) {
        MPI_Irecv(tmp_image[startrow-1], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &startrequest);
        MPI_Isend(tmp_image[endrow-1], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &request);
      }
      if (myrank == MASTER) {
        MPI_Irecv(tmp_image[endrow], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &endrequest);
      }
      else if (myrank == size-1) {
        MPI_Isend(tmp_image[startrow], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &request);
      }
      else if (myrank%2 == 0){
        MPI_Irecv(tmp_image[endrow], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &endrequest);
        MPI_Isend(tmp_image[startrow],ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &request);
      }
      else if (myrank%2 == 1){
        MPI_Isend(tmp_image[startrow],ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &request);
        MPI_Irecv(tmp_image[endrow], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &endrequest);
      }
      if (myrank != MASTER) MPI_Wait(&startrequest, &status);
      if (myrank != size-1) MPI_Wait(&endrequest, &status);
    }
    // Run stencil on same section of board
    stencil(startrow, endrow, nx, ny, tmp_image, image);
    // Send/receive first and last rows to neighboring processes.
    if (size > 1 && t != niters-1) {
      if (myrank == MASTER) {
        MPI_Isend(image[endrow-1],ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &request);
      }
      else if (myrank == size-1) {
        MPI_Irecv(image[startrow-1], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &startrequest);
      }
      else if (myrank%2 == 0){
        MPI_Isend(image[endrow-1],ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &request);
        MPI_Irecv(image[startrow-1], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &startrequest);
      }
      else if (myrank%2 == 1) {
        MPI_Irecv(image[startrow-1], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &startrequest);
        MPI_Isend(image[endrow-1],ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &request);
      }
      if (myrank == MASTER) {
        MPI_Irecv(image[endrow], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &endrequest);
      }
      else if (myrank == size-1) {
        MPI_Isend(image[startrow], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &request);
      }
      else if (myrank%2 == 0) {
        MPI_Irecv(image[endrow], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &endrequest);
        MPI_Isend(image[startrow], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &request);
      }
      else if (myrank%2 == 1) {
        MPI_Isend(image[startrow], ny, MPI_FLOAT, myrank-1, 0, MPI_COMM_WORLD, &request);
        MPI_Irecv(image[endrow], ny, MPI_FLOAT, myrank+1, 0, MPI_COMM_WORLD, &endrequest);
      }
      if (myrank != MASTER) MPI_Wait(&startrequest, &status);
      if (myrank != size-1) MPI_Wait(&endrequest, &status);
    }
  }
  double toc = wtime();

  if (myrank == MASTER) {
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    // Receive sections of final board from other processes.
    for (int source = 1; source < size; ++source) {
      for (int i = calcStart(nx, source, size); i < calcEnd(nx, source, size); ++i) {
        MPI_Recv(image[i], ny, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    printf("All sections received by master! Outputting image...\n");
    output_image(OUTPUT_FILE, nx, ny, image);
    free(image);
  }

  else {
    //printf("PROCESS %d: sending final section to master...\n", myrank);
    for (int i = startrow; i < endrow; ++i) {
      MPI_Send(image[i],ny, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
      //printf("Process %d: row %d sent!\n", myrank, i);
    }
    //printf("PROCESS %d: final section sent to master!\n", myrank);
  }

  // Finialise the MPI enviroment abd exit the program.
  MPI_Finalize();
  return EXIT_SUCCESS;
}

// Calculate start row based on rank
int calcStart(int nx, int rank, int size) {
  int startrow = rank * nx/size;
  return startrow;
}

// Calculate end row based on rank
int calcEnd(int nx, int rank, int size) {
  int endrow = (rank+1) * nx/size;
  return endrow;
}

void stencil(const int startrow, const int endrow, const int nx, const int ny, float **restrict image, float **restrict tmp_image) {
  // FIRST ROW OF SECTION
  // Check if first row of section is the first row of the image
  if (startrow == 0) {
    tmp_image[0][0] = image[0][0] * 0.6f + (image[1][0] + image[0][1]) * 0.1f;
    for (int j = 1; j < ny-1; ++j) {
      tmp_image[0][j] = image[0][j] * 0.6f + (image[1][j] + image[0][j-1] + image[0][j+1]) * 0.1f;
    }
    tmp_image[0][ny-1] = image[0][ny-1] * 0.6f + (image[1][ny-1] + image[0][ny-2]) * 0.1f;
  }
  else {
    tmp_image[startrow][0] = image[startrow][0] * 0.6f + (image[startrow+1][0] + image[startrow-1][0] + image[startrow][1]) * 0.1f;
    for (int j = 1; j < ny-1; ++j) {
      tmp_image[startrow][j] = image[startrow][j] * 0.6f + (image[startrow+1][j] + image[startrow-1][j] + image[startrow][j-1] + image[startrow][j+1]) * 0.1f;
    }
    tmp_image[startrow][ny-1] = image[startrow][ny-1] * 0.6f + (image[startrow+1][ny-1] + image[startrow-1][ny-1] + image[startrow][ny-2]) * 0.1f;
  }
  // MIDDLE ROWS OF SECTION
  for (int i = startrow+1; i < endrow-1; ++i) {
    tmp_image[i][0] = image[i][0] * 0.6f + (image[i-1][0] + image[i+1][0] + image[i][1]) * 0.1f;
    for (int j = 1; j < ny-1; ++j) {
      tmp_image[i][j] = image[i][j] * 0.6f + (image[i-1][j] + image[i+1][j] + image[i][j-1] + image[i][j+1]) * 0.1f;
    }
    tmp_image[i][ny-1] = image[i][ny-1] * 0.6f + (image[i-1][ny-1] + image[i+1][ny-1] + image[i][ny-2]) * 0.1f;
  }
  // BOTTOM ROW OF SECTION
  // Check if bttom row of section is the bottom row of the image
  if (endrow == nx) {
    tmp_image[endrow-1][0] = image[endrow-1][0] * 0.6f + (image[endrow-2][0] + image[endrow-1][1]) * 0.1f;
    for (int j = 1; j < ny-1; ++j) {
      tmp_image[endrow-1][j] = image[endrow-1][j] * 0.6f + (image[endrow-2][j] + image[endrow-1][j-1] + image[endrow-1][j+1]) * 0.1f;
    }
    tmp_image[endrow-1][ny-1] = image[endrow-1][ny-1] * 0.6f + (image[endrow-2][ny-1] + image[endrow-1][ny-2]) * 0.1f;
  }
  else {
    tmp_image[endrow-1][0] = image[endrow-1][0] * 0.6f + (image[endrow][0] + image[endrow-2][0] + image[endrow-1][1]) * 0.1f;
    for (int j = 1; j < ny-1; ++j) {
      tmp_image[endrow-1][j] = image[endrow-1][j] * 0.6f + (image[endrow][j] + image[endrow-2][j] + image[endrow-1][j-1] + image[endrow-1][j+1]) * 0.1f;
    }
    tmp_image[endrow-1][ny-1] = image[endrow-1][ny-1] * 0.6f + (image[endrow][ny-1] + image[endrow-2][ny-1] + image[endrow-1][ny-2]) * 0.1f;
  }
}

// Create the input image
void init_image(const int nx, const int ny, float **  image, float **  tmp_image) {
  // Zero everything
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      image[i][j] = 0.0f;
      tmp_image[i][j] = 0.0f;
    }
  }

  // Checkerboard
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
        for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
          if ((i+j)%2)
          image[ii][jj] = 100.0f;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float **  image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0f;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      if (image[i][j] > maximum)
        maximum = image[i][j];
    }
  }

  // Output image, converting to numbers 0-255
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      fputc((char)(255.0f*image[i][j]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
