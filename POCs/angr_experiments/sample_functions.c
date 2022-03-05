#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>


void swap(int *xp, int *yp) 
{ 
    int temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 

void bubbleSort3(int arr[], int n) 
{ 
   int i, j; 
   for (i = 0; i < n-1; i++)
  
       // Last i elements are already in place    
       for (j = 0; j < n-i-1; j++)  
           if (arr[j] > arr[j+1]) 
              swap(&arr[j], &arr[j+1]); 
} 

int isSorted(int arr[], int n)
{
	int i;
	for (i = 0; i < n - 1; i++)
		if (arr[i] > arr[i+1])
			return 0;
	return 1;
}

int getMax(int arr[], int n)
{
	int i;
	int max = 0;
	for (i = 0; i < n; i++)
		if (arr[i] > max)
			max = arr[i];
	return max;
}

float getAverage(float arr[], int n)
{
	float sum = 0.0;
	int i;
	for (i = 0; i < n; i++)
		sum += arr[i];
	return sum / n;
}

float square(float x )
{
    float p;
    p = x * x;
    return p;
}


int main(int argc, char **argv)
{
	return 0;
}