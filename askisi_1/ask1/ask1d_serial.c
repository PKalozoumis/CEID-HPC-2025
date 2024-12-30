#include <zfp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

size_t compress_array(float *data, size_t N, unsigned char **compressedData)
{
  // int precision: Affects truncation. Default is 24 bits. Cannot exceed 32 bits for floats
  // double tolerance: The accuracy. How close the compressed data is to the original. Controls the MAX ERROR allowed during the compression process. Default is 0

  for (int i = 0; i < N; i++)
  {
    printf("original data[%d]: %f\n", i, data[i]);
  }

  zfp_type type = zfp_type_float; // Specify float type
  zfp_field *field = zfp_field_1d(data, type, N);
  if (field == NULL)
  {
    fprintf(stderr, "ZFP field creation failed\n");
    exit(EXIT_FAILURE);
  }

  zfp_stream *zfp = zfp_stream_open(NULL);
  if (zfp == NULL)
  {
    fprintf(stderr, "ZFP stream creation failed\n");
    exit(EXIT_FAILURE);
  }

  zfp_stream_set_accuracy(zfp, 0.001);

  // Allocate buffer for compressed data
  size_t bufsize = zfp_stream_maximum_size(zfp, field);

  *compressedData = (unsigned char *)malloc(bufsize);
  if (*compressedData == NULL)
  {
    printf("Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // Associate buffer with ZFP stream
  bitstream *stream = stream_open(*compressedData, bufsize);
  if (stream == NULL)
  {
    fprintf(stderr, "Bitstream creation failed\n");
    free(compressedData); // Free allocated memory
    zfp_field_free(field);
    zfp_stream_close(zfp);
    exit(EXIT_FAILURE);
  }

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  // Compress
  size_t compressedSize = zfp_compress(zfp, field);
  if (compressedSize == 0)
  {
    fprintf(stderr, "ZFP compression failed\n");
    free(compressedData);
    stream_close(stream);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    exit(EXIT_FAILURE);
  }

  printf("Compressed size: %zu\n", compressedSize);

  // Clean up
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);

  return compressedSize;
}

//============================================================================================

void decompress_array(unsigned char *compressedData, size_t compressedSize, float *originalData, size_t N)
{
  zfp_type type = zfp_type_float;

  zfp_field *field = zfp_field_1d(originalData, type, N);
  if (field == NULL)
  {
    fprintf(stderr, "Failed to create ZFP field for decompression\n");
    exit(EXIT_FAILURE);
  }

  zfp_stream *zfp = zfp_stream_open(NULL);
  if (zfp == NULL)
  {
    fprintf(stderr, "Failed to create ZFP stream for decompression\n");
    zfp_field_free(field); // Free the field before exiting
    exit(EXIT_FAILURE);
  }

  zfp_stream_set_accuracy(zfp, 0.001);

  size_t bufsize = zfp_stream_maximum_size(zfp, field);

  bitstream *stream = stream_open(compressedData, bufsize);
  if (stream == NULL)
  {
    fprintf(stderr, "Failed to create bitstream for decompression\n");
    zfp_field_free(field); // Free the field before exiting
    zfp_stream_close(zfp); // Close the stream before exiting
    exit(EXIT_FAILURE);
  }
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  if (!zfp_decompress(zfp, field))
  {
    fprintf(stderr, "ZFP decompression failed\n");
    stream_close(stream);  // Close the bitstream before exiting
    zfp_field_free(field); // Free the field before exiting
    zfp_stream_close(zfp); // Close the stream before exiting
    exit(EXIT_FAILURE);
  }
  
  for (int i = 0; i < N; i++) // Print first 10 values for quick verification
  {
    printf("Decompressed data[%d]: %f\n", i, originalData[i]);
  }

    // Clean up
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
}

//================================================================

int main(int argc, char *argv[])
{

  int errFlag = 0;
  int *flags;

  // Argument parsing
  //============================================================================================
  if (argc != 2)
  {
    printf("Give size\n");
    return 0;
  }

  int N = atoi(argv[1]);

  int arraySize = N;

  //============================================================================================

  int outdata = 0;

  printf("Start size: %ld\n\n\n", arraySize * sizeof(float));

  // Parallel region
  //============================================================================================

  float *data = (float *)malloc(arraySize * sizeof(float));
  if (data == NULL)
  {
    printf("Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the matrix
  //============================================================================================
  srand(time(NULL));

  for (int i = 0; i < arraySize; i++)
  {
    data[i] = (rand() / (float)RAND_MAX) * 1000;
  }


  // Compress the matrix
  //============================================================================================

  unsigned char *compressedData = NULL;
  size_t compressedSize = compress_array(data, N, &compressedData);

  float *decompressedData = (float *)malloc(arraySize * sizeof(float));
  if (decompressedData == NULL)
  {
    printf("Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  decompress_array(compressedData, compressedSize, decompressedData, N);
  // printf("decompressed: %lf, data: %lf, arraysize: %d\n", decompressedData[4], data[4], arraySize);

  for (int i = 0; i < arraySize; i++)
  {
    if (fabs(decompressedData[i] - data[i]) > 2)
    {
      printf("decompressed: %lf, data: %lf, i: %d\n", decompressedData[i], data[i], i);
      errFlag = 1;

      break;
    }
  }

  // free(data);
  free(compressedData);
  free(decompressedData);

  // End of parallel region

  flags = (int *)malloc(sizeof(int));

  int success = 1;
  for (int i = 0; i < 1; i++)
  {
    if (flags[i] == 1)
    {
      printf("Verification failed.\n");
      success = 0;
      break;
    }
  }

  free(flags);

  if (success)
  {
    printf("Successful verification.\n");
  }

  return 0;
}