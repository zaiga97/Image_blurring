#!/bin/bash
mpicc -fopenmp -o blur.mpi_omp blur.mpi_omp.c -lm
