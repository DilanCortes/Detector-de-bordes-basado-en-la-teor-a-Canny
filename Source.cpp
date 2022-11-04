/* Examen Practico Visión Artificial
 Cortes Martinez Dilan
 5BV1
 Escuela Superior de Computo
*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cmath>


using namespace cv;
using namespace std;

//Funciones creada para la creacion del Kernel Gaussiano
float** CrearKernelGauss(float** Kernel, int d);
float** LlenarKernelGauss(float** Kernel, int d);
float Gauss(int x, int y);

//Funcion creada para la creacion de los Kernel de Sobel
float** CrearKernelSobel(float** Kernel, int Ksize, char eje);

//Funciones para manipular los Kernel
void ImprimirKernel(float** Kernel, int d, char filtro);
void DestruirKernel(float** Kernel, int d, char filtro);

//Funciones para la aplicacion de filtros
float AplicarFiltroAlPixel(Mat image, float** Kernel, int Ksize, int i, int j);
Mat AplicarfiltroGauss(Mat imagen, float** Kernel, int Tamañokernel);
Mat EliminarBordes(Mat imagen, int Tamañokernel, char filtro);
Mat ConversionNTSC(Mat imagen);
Mat CrearImagenTransicion(Mat imagen, int Tamañokernel, char filtro);
Mat ecualizador(Mat imagen);
Mat AplicarfiltroSobel(Mat image, float** KSobel, char eje);
Mat AplicarGsobel(Mat imagen_1, Mat imagen_2);
Mat Umbral(Mat imagen);

//Funcion creada para mostrar las dimenciones de cada imagen
void ImprimirDimensiones(Mat imagen);



int main(int argc, char* argv[]) {
	float** KernelGauss = NULL;	//Declaracion del Kernel Gaussiano
	float** Sobelx = NULL;	//Declaracion del Kernel de Sobel para el eje x
	float** Sobely = NULL;	//Declaraciond el kernel de Sobel para el eje y
	int Ksize = 0, sigma = 0;	//Declaracion de los parametros necesarios para el Kernel Gaussiano

	printf("Introduzca el tamaño de su Kernel Gaussiano: ");
	scanf_s("%d", &Ksize);

	printf("Introduzca el tamaño de su Kernel Gaussiano: ");
	scanf_s("%d", &sigma);

	if (Ksize % 2 == 0) {	//Verificacion de que el kernel Gaussiano tiene una longitud impar
		
		printf("\nLa longitud del kernel Gaussiano debe ser impar, prueba otra vez\n");
			
		return(1);
	}
	else {
		printf("Kernel de Gauss (Dinamico)\n");
		KernelGauss = CrearKernelGauss(KernelGauss, Ksize);	//Creacion del Kernel Gaussiano con los paramteros indicados
		ImprimirKernel(KernelGauss, Ksize, 'G');

		printf("\nKernel de Sobel (Eje X)\n");
		Sobelx = CrearKernelSobel(Sobelx, 3, 'X');	//Creacion del Kernel Sobel para el eje X
		ImprimirKernel(Sobelx, Ksize, 'S');

		printf("\nKernel de Sobel (Eje Y)\n");
		Sobely = CrearKernelSobel(Sobely, 3, 'Y');	//Creacion del Kernel Sobel para el eje Y
		ImprimirKernel(Sobely, Ksize, 'S');

		printf("\nImagen original");
		Mat Imagen = imread("Lena.jpg");		//Cargamos la imagen con la que vamos a trabajar 'Lena.jpg'
		imshow("Imagen Original", Imagen);
		ImprimirDimensiones(Imagen);

		printf("\nImagen NTSC");
		Mat ImagenNTSC = ConversionNTSC(Imagen);	//Reazlizamos la conversion de la imagen a escala de grises
		imshow("Imagen NTSC", ImagenNTSC);
		ImprimirDimensiones(ImagenNTSC);

		Mat Imagentransicion_G = CrearImagenTransicion(ImagenNTSC, Ksize, 'G');	//Agregamos la cantidad de bordes necesarios a la imagen NTSC, para la aplicacion del Kernel Gaussiano 

		Mat ImagenGaussBordes = AplicarfiltroGauss(Imagentransicion_G, KernelGauss, Ksize);	//Aplicamos el filtro Gaussiano a la imagen con bordes adicionales

		printf("\nImagen Gauss");
		Mat ImagenFinalGauss = EliminarBordes(ImagenGaussBordes, Ksize, 'G');	//Aplicamos una eliminacion de  bordes a la imagen resultante
		imshow("Imagen filtrada sin bordes", ImagenFinalGauss);
		ImprimirDimensiones(ImagenFinalGauss);

		printf("\nImagen Ecualizada");
		Mat ImagenEcualizada = ecualizador(ImagenFinalGauss);
		imshow("Imagen Ecualizada", ImagenEcualizada);
		ImprimirDimensiones(ImagenEcualizada);

		Mat Imagentransicion_S = CrearImagenTransicion(ImagenEcualizada, Ksize, 'S');	//Agregamos la cantidad de bordes necesarios a la con el filtro Gaussiano, para la aplicacion de los Kernel de Sobel 

		Mat Imagen_SobelX = AplicarfiltroSobel(Imagentransicion_S, Sobelx, 'X');		//Aplicamos el Kernel de Sobel en el eje X

		Mat Imagen_SobelY = AplicarfiltroSobel(Imagentransicion_S, Sobely, 'Y');	//Aplicamos el Kernel de Sobel en el eje Y

		Mat G = AplicarGsobel(Imagen_SobelX, Imagen_SobelY);	//Aplicamos la formula para obtener |G| a partir de las imagenes previamente creadas con los filtros de sobel

		printf("\nImagen |G|");
		Mat G_F = EliminarBordes(G, Ksize, 'S');	//Aplicamos una eliminacion de  bordes a la imagen resultante
		imshow("Imagen |G| sin bordes", G_F);
		ImprimirDimensiones(G_F);

		printf("\nImagen Umbralizada");
		Mat Umbralado = Umbral(G_F);
		imshow("Imagen Umbralizada", Umbralado);
		ImprimirDimensiones(Umbralado);

		waitKey(0);	//Condicion de paro para visualizar las imagenes, una vez termine precione 0 para finalizar el programa
		DestruirKernel(KernelGauss, Ksize, 'G');
		DestruirKernel(Sobelx, Ksize, 'S');
		DestruirKernel(Sobely, Ksize, 'S');

		return(0);
	}
	
}

/*Es importante destacar que algunas funciones se usan tanto para Gauss como para Sobel, por lo tanto se establecio una variable char la cual funciona como llave de activacion
si esta es 'G', la dimension que se manjeara sera la del kernel dimanico (Gaussino), si no la dimension sera 3, por los kernel estaticos (Sobel). La misma logica se implemento para
las funciones de Sobel, si la llave es 'X' quiere decir que estamos trabajando con el kernel del eje x, de lo contrario se trabajara con el kernel del eje y*/

//Recibe como parametros un apuntador doble vacio (Kernel), el tamaño del kernel que se desea y devuelve el kernel Gaussiano ya creado de manera dinamica
float** CrearKernelGauss(float** Kernel, int d) 
{
	int i = 0, j = 0;

	Kernel = (float**)malloc(d * sizeof(float*));

	for (i = 0; i < d; i++)
		Kernel[i] = (float*)malloc(d * sizeof(float));

	Kernel = LlenarKernelGauss(Kernel, d);
	return (Kernel);
} 

//Recibe como parametros un apuntador doble vacio (Kernel) y el tamaño del kernel que se desea y lo rellena a partir de la division del kernel en 4 cuadrantes con la formula del kernel gaussiano
float** LlenarKernelGauss(float** Kernel, int d)
{
	//i representa el eje Y y j representa el eje X 

	int i = 0, j = 0;
	int x = 0, y = 0;

	for (i = d / 2; i < d; i++)	//Llenamos el cuadrante I
	{
		for (j = d / 2; j < d; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = 0;
		y += 1;
	}
	x = 0;
	y = -(d / 2);

	for (i = 0; i < d / 2; i++)	//Llenamos el cuadrante II
	{
		for (j = d / 2; j < d; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = 0;
		y += 1;
	}
	x = -(d / 2);
	y = -(d / 2);

	for (i = 0; i < d / 2; i++)	//Llenamos el cuadrante III
	{
		for (j = 0; j < d / 2; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = -(d / 2);
		y += 1;
	}
	x = -(d / 2);
	y = 0;

	for (i = d / 2; i < d; i++)	//Llenamos el cuadrante IV
	{
		for (j = 0; j < d / 2; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = -(d / 2);
		y += 1;
	}
	return(Kernel);
}

//Formula matematica que permite obtener el valor correspendiente para el kernel de Gauss, de un par de coordenadas.
float Gauss(int x, int y)
{
	float pi = 3.1416, e = 2.71828;
	float sigma = 1, F_1 = 0, F_2 = 0, potencia = 0;
	float valor = 0;

	F_1 = (1) / (2 * pi * pow(sigma, 2));
	potencia = (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2));
	F_2 = pow(e, 0 - potencia);
	valor = F_1 * F_2;

	return(valor);
}

//Recibe como parametros un apuntador doble vacio (Kernel), el tamaño del kernel y la llave del eje correspondiente al Kernel que se desea y devuelve el kernel Sobel correspondiente ya creado
float** CrearKernelSobel(float** Kernel, int Ksize, char eje) {

	int i = 0, j = 0;

	Kernel = (float**)malloc(Ksize * sizeof(float*));

	for (i = 0; i < Ksize; i++)
		Kernel[i] = (float*)malloc(Ksize * sizeof(float));
	if (eje == 'X') {
		for (i = 0; i < Ksize; i++)
			for (j = 0; j < Ksize; j++)
				if (j == 1) {
					Kernel[i][j] = 0;
				}
				else if (j == 0 && (i == 0 || i == 2)) {
					Kernel[i][j] = -1;
				}
				else if (j == 0 && i == 1) {
					Kernel[i][j] = -2;
				}
				else if (j == 2 && (i == 0 || i == 2)) {
					Kernel[i][j] = 1;
				}
				else {
					Kernel[i][j] = 2;
				}
	}
	else {
		for (i = 0; i < Ksize; i++)
			for (j = 0; j < Ksize; j++)
				if (i == 1) {
					Kernel[i][j] = 0;
				}
				else if (i == 0 && (j == 0 || j == 2)) {
					Kernel[i][j] = 1;
				}
				else if (i == 0 && j == 1) {
					Kernel[i][j] = 2;
				}
				else if (i == 2 && (j == 0 || j == 2)) {
					Kernel[i][j] = -1;
				}
				else {
					Kernel[i][j] = -2;
				}
	}
	return (Kernel);
}

//Recibe como parametros un apuntador doble (Kernel), su dimension y la llave del filtro al que pertenece, imprime el kernel 
void ImprimirKernel(float** Kernel, int d, char filtro)
{
	if (filtro == 'S') {
		d = 3;
		int i = 0, j = 0;

		for (i = 0; i < d; i++)
		{
			for (j = 0; j < d; j++)
				printf("%.0f\t", Kernel[i][j]);
			printf("\n");
		}
	}
	else {
		int i = 0, j = 0;

		for (i = 0; i < d; i++)
		{
			for (j = 0; j < d; j++)
				printf("%.3f\t", Kernel[i][j]);
			printf("\n");
		}
	}

}

//Recibe como parametros un apuntador doble (Kernel), su dimension y la llave del filtro al que pertenece, libera la memoria correspondiente al Kernel
void DestruirKernel(float** Kernel, int d, char filtro)
{
	if (filtro == 'S') {
		d = 3;
	}
	int i = 0, j = 0;
	for (i = 0; i < d; i++)
	{
		free(Kernel[i]);
		Kernel[i] = NULL;
	}
	free(Kernel);
}

/*Recibe como paramteros una imagen en formato Mat, el kernel a aplicar, el tamaño del kernel y un par de coordenadas x->i; y->j.La funcion detecta los vecinos correspondientes por medio del kernel
del pixel que se le envia, calcula los productos correspnodiente, realiza las sumatorias y retorna el nuevo valor correspndiente al pixel*/
float AplicarFiltroAlPixel(Mat image, float** Kernel, int Ksize, int x, int y) {
	int rows = image.rows;
	int cols = image.cols;
	int amountSlide = (Ksize - 1) / 2;
	float sumFilter = 0;
	float sumKernel = 0;
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			float valork = Kernel[i + amountSlide][j + amountSlide];
			int X = x + i;
			int Y = y + j;
			float valor = 0;
			if (!(X < 0 || X >= cols || Y < 0 || Y >= rows)) {
				valor = image.at<uchar>(Point(Y, X));
			}

			sumFilter += (valork * valor);
			sumKernel += valork;
		}
	}
	return sumFilter / sumKernel;
}

/*Recibe como parametros una imagen en fomrato Mat, un kernel y el tamaño del kernel, recorre la imagen pixel por pixel exceptuando los bordes auxiliares.Invoca a AplicarFiltroAlPixel
retorna una imagen ya filtrada en formato Mat*/
Mat AplicarfiltroGauss(Mat imagen, float** Kernel, int Tamañokernel) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	int exceso = (Tamañokernel - 1);
	int exceso2 = exceso / 2;

	Mat ImagenFiltrada(rows, cols, CV_8UC1);
	double valor;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			if (i >= rows - exceso2 || i < exceso2) {
				ImagenFiltrada.at<uchar>(Point(j, i)) = uchar(0);
			}
			else if (j >= cols - exceso2 || j < exceso2) {
				ImagenFiltrada.at<uchar>(Point(j, i)) = uchar(0);
			}
			else {
				ImagenFiltrada.at<uchar>(Point(j, i)) = uchar(AplicarFiltroAlPixel(imagen, Kernel, Tamañokernel, i, j));
			}

		}
	}
	return(ImagenFiltrada);
}

/*Recibe como parametros una imagen en formato Mat, un kernel y la llave que indique a que eje pertenece el kernel, recorre la imagen pixel por pixel exceptuando los bordes auxiliares.Invoca a AplicarFiltroAlPixel
 retorna una imagen ya filtrada en formato Mat*/
Mat AplicarfiltroSobel(Mat image, float** KSobel, char eje) {
	int rows = image.rows;
	int cols = image.cols;
	int Tamañokernel = 3;
	int exceso = (Tamañokernel - 1);
	int exceso2 = exceso / 2;

	Mat sobel(rows, cols, CV_8UC1);
	double valor = 0;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			if (i >= rows - exceso2 || i < exceso2) {
				sobel.at<uchar>(Point(j, i)) = uchar(0);
			}
			else if (j >= cols - exceso2 || j < exceso2) {
				sobel.at<uchar>(Point(j, i)) = uchar(0);
			}
			else {
				sobel.at<uchar>(Point(j, i)) = uchar(AplicarFiltroAlPixel(image, KSobel, Tamañokernel, i, j));
			}

		}
	}

	if (eje == 'X') {
		Sobel(image, sobel, CV_8U, 1, 0, 1, 1, 0, BORDER_DEFAULT);
	}
	else {
		Sobel(image, sobel, CV_8U, 0, 1, 1, 1, 0, BORDER_DEFAULT);
	}
	return(sobel);
}

//Recibe commo parametros una imagen en formato Mat y devueve una imagen ecualizada por medio de su histograma en formato Mat
Mat ecualizador(Mat imagen) {
	Mat auxiliar;
	equalizeHist(imagen,auxiliar);
	return(auxiliar);
}

//Recibe como parametros una imagen en formato Mat, el tamaño del kernel aplicado a la imagen y la llave que indica el filtro que se aplico, devuelve la imagen sin los bordes auxiliares
Mat EliminarBordes(Mat imagen, int Tamañokernel, char filtro) {

	if (filtro == 'S') {
		Tamañokernel = 3;
	}
	int rows = imagen.rows;
	int cols = imagen.cols;
	int exceso = (Tamañokernel - 1);
	int exceso2 = exceso / 2;

	Mat Normal(rows - exceso, cols - exceso, CV_8UC1);
	double valor;

	for (int i = 0; i < rows - exceso; i++) {
		for (int j = 0; j < cols - exceso; j++) {
			valor = imagen.at<uchar>(i + exceso2, j + exceso2);
			Normal.at<uchar>(Point(j, i)) = uchar(valor);

		}
	}
	return(Normal);
}

//Recibe como parametro una imagen en formato Mat A COLOR, y devuelve la imagen en escala de grises por medio del metodo NTSC en formato Mat
Mat ConversionNTSC(Mat imagen) {
	int rows = imagen.rows;
	int cols = imagen.cols;

	Mat imagenGrisesNTSC(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double azul = imagen.at<Vec3b>(Point(j, i)).val[0];  // B
			double verde = imagen.at<Vec3b>(Point(j, i)).val[1]; // G
			double rojo = imagen.at<Vec3b>(Point(j, i)).val[2];  // R

			// Conversion a escala de grises
			imagenGrisesNTSC.at<uchar>(Point(j, i)) = uchar(0.299 * rojo + 0.587 * verde + 0.114 * azul);
		}
	}
	return(imagenGrisesNTSC);
}

//Recibe como parametos una imagen en formato Mat, el tamaño del kernel que se le va a plicar y la llave que indica el filtro que se le va a aplicar, devuelve una imagen en formato Mat con los bordes auxiliares 
Mat CrearImagenTransicion(Mat imagen, int Tamañokernel, char filtro) {
	if (filtro == 'S') {
		Tamañokernel = 3;
	}
	int rows = imagen.rows;
	int cols = imagen.cols;
	int exceso = (Tamañokernel - 1);
	int exceso2 = exceso / 2;

	Mat transicion(rows + exceso, cols + exceso, CV_8UC1);
	double valor;

	for (int i = 0; i < rows + exceso; i++) {
		for (int j = 0; j < cols + exceso; j++) {

			if (i >= rows + exceso2 || i < exceso2) {
				transicion.at<uchar>(Point(j, i)) = uchar(0);
			}
			else if (j >= cols + exceso2 || j < exceso2) {
				transicion.at<uchar>(Point(j, i)) = uchar(0);
			}
			else {

				transicion.at<uchar>(Point(j, i)) = uchar(imagen.at<uchar>(i - exceso2, j - exceso2));
			}

		}
	}
	return(transicion);
}

//Recibe como paramteros dos imagenes en formato Mat correspondientes a los ejer X y Y del filtro de sobel, retorna una imagen en formato Mat con los valores correspondientes para cada pixel de la formula |G|
Mat AplicarGsobel(Mat imagen_1, Mat imagen_2) {
	int rows = imagen_1.rows;
	int cols = imagen_2.cols;

	Mat G(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			G.at<uchar>(Point(j, i)) = sqrt(pow(imagen_1.at<uchar>(i,j), 2) + pow(imagen_2.at<uchar>(i, j), 2));
		}
	}
	return(G);
}

//Recibe una imagen convertida a Mat y permite obtener e imprimir las dimensiones correspondientes
void ImprimirDimensiones(Mat imagen) {
	int fila_original = 0, columna_original = 0;
	fila_original = imagen.rows;
	columna_original = imagen.cols;
	printf("\n%d pixeles de largo \n", fila_original);
	printf("%d pixeles de ancho\n\n", columna_original);

}

//Recibe como parametro una imagen en formato Mat y devuelve una imagen umbralizada en formato Mat
Mat Umbral(Mat imagen)
{
	Mat Recipiente;
	int umbral_valor = 12;
	int max_binary_value = 255;
	int umbral_tipo = 0;
	threshold(imagen, Recipiente, umbral_valor, max_binary_value, umbral_tipo);
	return Recipiente;
}