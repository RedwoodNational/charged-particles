
#include <iostream>
#include <fstream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

#include <cmath>
#include <vector>
#include <random>

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err)
			<< " in " << std::string(file)
			<< " at line " << line << "\n";
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

std::random_device rd;      // генератор случайных чисел
std::mt19937 mersenne(rd());// алгоритм мерсенна
{{
int a         = 15;    // половина стороны куба   
int n         = 150;   // количество частиц  
float eps     = 1e-3;  // избежать деление на ноль  
float wc      = 0.99;  // коэффициент замедления  
float k       = 50.0;  // коэффициент пропорциональности  
float dt      = 0.01;  // шаг интегрирования  
float g       = 20;    // ускорение свободного падения  
float qi      = 1;     // заряды частиц  
float qc      = 30;    // заряд камеры  
float qb      = 50;    // заряд пули  
float vb      = 30;    // скорость пули  
float shift_z = 0.75;  // сдвиг карты напряженности

float radius = 1;
}}


std::vector<float> pq;

std::vector<float> px;  //x coords
std::vector<float> pxx; //temp buffers
std::vector<float> py;  //y coords
std::vector<float> pyy; //temp buffers
std::vector<float> pz;  //z coords
std::vector<float> pzz; //temp buffers

float* cpx = nullptr; //cuda buffers
float* cpy = nullptr; //cuda buffers
float* cpz = nullptr; //cuda buffers

float* cpq = nullptr; //cuda buffers

std::vector<float> pvx; //vx speed
std::vector<float> pvy; //vy speed
std::vector<float> pvz; //vz speed

int w        = 1024;  //width resolution
int h        = 648;   //height resolution

float xc     = -1.5; // x камеры
float yc     = -1.5; // y камеры
float zc     = 1.0;  // z камеры
float dx     = 0.0; 
float dy     = 0.0; 
float dz     = 0.0; 

float xb = -1000000; // x пули
float yb = -1000000; // y пули
float zb = 1000000;  // z пули

float vxb = -1.5; // vx пули
float vyb = -1.5; // vy пули
float vzb = 1.0;  // vz пули



float yaw    = 0.0; // рыскание
float pitch  = 0.0; // тангаж
float dyaw   = 0.0;
float dpitch = 0.0;


#define M_PI 3.14f
typedef unsigned char uchar;

float speed = 0.2;


const int np = 100;				// Размер текстуры пола


GLUquadric* quadric;			// quadric объекты - это геометрические фигуры 2-го порядка, т.е. сфера, цилиндр, диск, конус. 

cudaGraphicsResource* res;
GLuint textures[2];				// Массив из текстурных номеров
GLuint vbo;						// Номер буфера


__global__ void kernel(uchar4* data, int n, 
	float a, float k, float eps, float sz, int pn,
	float* px, float* py, float *pz, float* pq,
	float  xb, float  yb, float zb, float  qb ) {	// Генерация текстуры пола на GPU

	int offset = blockDim.x * gridDim.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int size = n * n;

	float x;
	float y;
	float e = 0.0;
	int p;
	while (i < size) {

		x = a * (2.0 * static_cast<float>(i % n) 
			/ static_cast<float>(n) - 1);
		y = a * (2.0 * static_cast<float>(i / n) 
			/ static_cast<float>(n) - 1);
		e = qb / ( eps + 
			powf(xb - x, 2) +
			powf(yb - y, 2) +
			powf(zb - sz, 2)
		);
		for (p = 0; p < pn; ++p) {
			e += pq[p] / (eps +
				powf(px[p] - x, 2) +
				powf(py[p] - y, 2) +
				powf(pz[p] - sz, 2)
			);
		}
		e *= k;
		data[i].x = min(e, 255.0);
		data[i].y = min(e, 255.0);
		data[i].z = min(e, 255.0);
		data[i].w = 1;
		i += offset;
	}

}

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Задаем "объектив камеры"
	gluPerspective(90.0f, (GLfloat)w / (GLfloat)h, 0.1f, 100.0f);


	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	// Задаем позицию и направление камеры
	gluLookAt(xc, yc, zc,
		xc + cos(yaw) * cos(pitch),
		yc + sin(yaw) * cos(pitch),
		zc + sin(pitch),
		0.0f, 0.0f, 1.0f);

	glBindTexture(GL_TEXTURE_2D, textures[0]);	// Задаем текстуру


	static float angle = 0.0;

	for (size_t i = 0; i < n; ++i) {
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);	// Задаем координаты центра сферы
		glRotatef(angle, 0.0, 0.0, 1.0);    // Поворачиваем сферу на угол
		gluSphere(quadric, 1.0f, 32, 32);
		glPopMatrix();
	}
	angle += 0.15;



	glPushMatrix();
	glTranslatef(xb, yb, zb);
	gluSphere(quadric, 1.0f, 32, 32);
	glPopMatrix();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);	// Делаем активным буфер с номером vbo
	glBindTexture(GL_TEXTURE_2D, textures[1]);	// Делаем активной вторую текстуру
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)np, (GLsizei)np, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);	// Деактивируем буфер
	// Последний параметр NULL в glTexImage2D говорит о том что данные для текстуры нужно брать из активного буфера

	glBegin(GL_QUADS);			               // Рисуем пол
	glTexCoord2f(0.0, 0.0);
	glVertex3f(-a, -a, 0.0);

	glTexCoord2f(1.0, 0.0);
	glVertex3f(a, -a, 0.0);

	glTexCoord2f(1.0, 1.0);
	glVertex3f(a, a, 0.0);

	glTexCoord2f(0.0, 1.0);
	glVertex3f(-a, a, 0.0);
	glEnd();


	glBindTexture(GL_TEXTURE_2D, 0);			// Деактивируем текстуру

	// Отрисовка каркаса куба				
	glLineWidth(2);								// Толщина линий				
	glColor3f(0.5f, 0.5f, 0.5f);				// Цвет линий
	glBegin(GL_LINES);							// Последующие пары вершин будут задавать линии
	glVertex3f(-a, -a, 0.0);
	glVertex3f(-a, -a, 2.0 * a);

	glVertex3f(a, -a, 0.0);
	glVertex3f(a, -a, 2.0 * a);

	glVertex3f(a, a, 0.0);
	glVertex3f(a, a, 2.0 * a);

	glVertex3f(-a, a, 0.0);
	glVertex3f(-a, a, 2.0 * a);
	glEnd();

	glBegin(GL_LINE_LOOP);						// Все последующие точки будут соеденены замкнутой линией
	glVertex3f(-a, -a, 0.0);
	glVertex3f(a, -a, 0.0);
	glVertex3f(a, a, 0.0);
	glVertex3f(-a, a, 0.0);
	glEnd();

	glBegin(GL_LINE_LOOP);
	glVertex3f(-a, -a, 2.0 * a);
	glVertex3f(a, -a, 2.0 * a);
	glVertex3f(a, a, 2.0 * a);
	glVertex3f(-a, a, 2.0 * a);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);

	glutSwapBuffers();
}



void update() {

	for (size_t i = 0; i < n; ++i) {
		float dvx = 0.0;
		float dvy = 0.0;
		float dvz = 0.0;
		float l = 0.0;
		// Отталкивание от частиц
		for (size_t j = 0; j < n; ++j) {
			l = std::pow(
					std::pow(px[i] - px[j], 2) +
					std::pow(py[i] - py[j], 2) + 
					std::pow(pz[i] - pz[j], 2), 
											1.5) + eps;

			dvx += pq[j] * (px[i] - px[j]) / l;
			dvy += pq[j] * (py[i] - py[j]) / l;
			dvz += pq[j] * (pz[i] - pz[j]) / l;
		}
		// Отталкивание от стен

		// правая стена
		l = eps + std::pow(std::abs(px[i] - a), 3);
		dvx += pq[i] * (px[i] - a) / l;
		// передняя стена
		l = eps + std::pow(std::abs(py[i] - a), 3);
		dvy += pq[i] * (py[i] - a) / l;
		// потолок
		l = eps + std::pow(std::abs(pz[i] - 2 * a), 3);
		dvz += pq[i] * (pz[i] - 2 * a) / l;

		// левая стена
		l = eps + std::pow(std::abs(px[i] + a), 3);
		dvx += pq[i] * (px[i] + a) / l;
		// задняя стена
		l = eps + std::pow(std::abs(py[i] + a), 3);
		dvy += pq[i] * (py[i] + a) / l;
		//пол
		l = eps + std::pow(std::abs(pz[i]), 3);
		dvz += pq[i] * (pz[i]) / l;



		// Отталкивание от камеры
		l = std::pow(	std::pow(px[i] - xc, 2) + 
						std::pow(py[i] - yc, 2) + 
						std::pow(pz[i] - zc, 2), 
											1.5) + eps;

		dvx += qc * (px[i] - xc) / l;
		dvy += qc * (py[i] - yc) / l;
		dvz += qc * (pz[i] - zc) / l;

		// Отталкивание от пули
		l = std::pow(	std::pow(px[i] - xb, 2) +
						std::pow(py[i] - yb, 2) +
						std::pow(pz[i] - zb, 2),
											1.5) + eps;

		dvx += qb * (px[i] - xb) / l;
		dvy += qb * (py[i] - yb) / l;
		dvz += qb * (pz[i] - zb) / l;


		// интегрируем

		dvx = k * pq[i] * dvx * dt;
		dvy = k * pq[i] * dvy * dt;
		dvz = k * pq[i] * dvz * dt -g * dt;

		pvx[i] = wc * pvx[i] + dvx;
		pvy[i] = wc * pvy[i] + dvy;
		pvz[i] = wc * pvz[i] + dvz;

		// сохранение импульса при ударе 
		if ((px[i] + pvx[i] * dt) >= a
		|| (px[i] + pvx[i] * dt) <= -a) {
			pvx[i] -= pvx[i];
		}

		if ((py[i] + pvy[i] * dt) >= a
		|| (py[i] + pvy[i] * dt) <= -a) {
			pvy[i] -= pvy[i];
		}

		if ((pz[i] + pvz[i] * dt) >= 2 * a
		||  (pz[i] + pvz[i] * dt) <= 0) {
			pvz[i] -= pvz[i];
		}
		// изменяем коодинаты
		pxx[i] = px[i] + pvx[i] * dt;
		pyy[i] = py[i] + pvy[i] * dt;
		pzz[i] = pz[i] + pvz[i] * dt;
	}

	std::swap(px, pxx);
	std::swap(py, pyy);
	std::swap(pz, pzz);

	// движение пули

	xb += vxb * dt;
	yb += vyb * dt;
	zb += vzb * dt;

	float v = std::sqrt(dx * dx + dy * dy + dz * dz);
	if (v > speed) {		// Ограничение максимальной скорости
		dx *= speed / v;
		dy *= speed / v;
		dz *= speed / v;
	}
	xc += dx; dx *= 0.95;
	yc += dy; dy *= 0.95;
	zc += dz; dz *= 0.95;

	if (std::abs(dpitch) + fabs(dyaw) > 0.0001) {	// Вращение камеры
		yaw   += dyaw;
		pitch += dpitch;
		pitch = std::min(M_PI / 2.0f - 0.0001f, std::max(-M_PI / 2.0f + 0.0001f, pitch));
		dyaw  = dpitch = 0.0;
	}




	uchar4* dev_data;
	size_t size;
	cudaGraphicsMapResources(1, &res, 0);		// Делаем буфер доступным для CUDA
	cudaGraphicsResourceGetMappedPointer((void**)&dev_data, &size, res);	// Получаем указатель на память буфера

	cudaMemcpy(cpx, px.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cpy, py.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cpz, pz.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cpq, pq.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	kernel <<<256, 128>> > (dev_data, np, a,  k, eps, shift_z, n, cpx, cpy, cpz, cpq, xb, yb, zb, qb);
	cudaGraphicsUnmapResources(1, &res, 0);		// Возращаем буфер OpenGL'ю что бы он мог его использовать


	glutPostRedisplay();	// Перерисовка
}

void keys(unsigned char key, int x, int y) {	// Обработка кнопок
	switch (key) {
	case 'w':                 // "W" Вперед
		dx = cos(yaw) * cos(pitch) * speed;
		dy = sin(yaw) * cos(pitch) * speed;
		dz = sin(pitch) * speed;
		break;
	case 's':                 // "S" Назад
		dx = -cos(yaw) * cos(pitch) * speed;
		dy = -sin(yaw) * cos(pitch) * speed;
		dz = -sin(pitch) * speed;
		break;
	case 'a':                 // "A" Влево
		dx = -sin(yaw) * speed;
		dy = cos(yaw) * speed;
		break;
	case 'd':                 // "D" Вправо
		dx = sin(yaw) * speed;
		dy = -cos(yaw) * speed;
		break;
	case 27:                 // "ESC" Выход
		cudaGraphicsUnregisterResource(res);
		glDeleteTextures(2, textures);
		glDeleteBuffers(1, &vbo);
		gluDeleteQuadric(quadric);

		HANDLE_ERROR(cudaFree(cpx));
		HANDLE_ERROR(cudaFree(cpy));
		HANDLE_ERROR(cudaFree(cpz));
		HANDLE_ERROR(cudaFree(cpq));
		exit(0);
		break;
	}
}

void mouse(int x, int y) {
	static int x_prev = w / 2;
	static int y_prev = h / 2;
	float dx = 0.005 * (x - x_prev);
	float dy = 0.005 * (y - y_prev);
	dyaw   -= dx;
	dpitch -= dy;
	x_prev = x;
	y_prev = y;

	// Перемещаем указатель мышки в центр, когда он достиг границы
	if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) {
		glutWarpPointer(w / 2, h / 2);
		x_prev = w / 2;
		y_prev = h / 2;
	}
}
void mouseClicks(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && GLUT_DOWN == state) {
		vxb = vb * cos(yaw) * cos(pitch);
		vyb = vb * sin(yaw) * cos(pitch);
		vzb = vb * sin(pitch);

		xb = xc + vxb * 3 * dt;
		yb = yc + vyb * 3 * dt;
		zb = zc + vzb * 3 * dt;
	}
}
void reshape(int w_new, int h_new) {
	w = w_new;
	h = h_new;
	glViewport(0, 0, w, h);                                     // Сброс текущей области вывода
	glMatrixMode(GL_PROJECTION);                                // Выбор матрицы проекций
	glLoadIdentity();                                           // Сброс матрицы проекции
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); // двойная буферизация
	glutInitWindowSize(w, h);
	glutCreateWindow("OpenGL");
	                                              // назначаем обработчики
	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(keys);
	glutMouseFunc(mouseClicks);
	glutPassiveMotionFunc(mouse);
	glutReshapeFunc(reshape);

	glutSetCursor(GLUT_CURSOR_NONE);	          // Скрываем курсор мышки


	px.resize(n);
	py.resize(n);
	pz.resize(n);


	pvx.resize(n);
	pvy.resize(n);
	pvz.resize(n);

	pxx.resize(n);
	pyy.resize(n);
	pzz.resize(n);

	pq.resize(n);

	for (size_t i = 0; i < n; ++i) {
		pq[i] = qi;
		pvx[i] = 0;
		pvy[i] = 0;
		pvz[i] = 0;
		px[i] = dist_x(mersenne);
		py[i] = dist_y(mersenne);
		pz[i] = dist_z(mersenne);
	}

	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&cpx), n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&cpy), n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&cpz), n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&cpq), n * sizeof(float)));

	int wt , ht;
	std::fstream in("in.data", std::ios::binary | std::ios::in);
	if (!in.is_open()) {
		std::cerr << "Could not open file\n";
		std::cerr << "Make sure in.data exists\n";
	}
	in.read(reinterpret_cast<char*>(&wt), sizeof(int));
	in.read(reinterpret_cast<char*>(&ht), sizeof(int));
	std::cout << wt << " " << ht << std::endl;
	uchar* data = reinterpret_cast<uchar*>(operator new(wt * ht * sizeof(int)));
	in.read(reinterpret_cast<char*>(data), wt * ht * sizeof(int));
	in.close();

	glGenTextures(2, textures);
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
	// если полигон, на который наносим текстуру, меньше текстуры
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //GL_LINEAR);	// Интерполяция
	// если больше
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //GL_LINEAR);		
	//operator delete(data);


	quadric = gluNewQuadric();
	gluQuadricTexture(quadric, GL_TRUE);

	glBindTexture(GL_TEXTURE_2D, textures[1]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	// Интерполяция 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	// Интерполяция	

	glEnable(GL_TEXTURE_2D);                             // Разрешить наложение текстуры
	glShadeModel(GL_SMOOTH);                             // Разрешение сглаженного закрашивания
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);                // Черный фон
	glClearDepth(1.0f);                                  // Установка буфера глубины
	glDepthFunc(GL_LEQUAL);                              // Тип теста глубины. 
	glEnable(GL_DEPTH_TEST);                			 // Включаем тест глубины
	glEnable(GL_CULL_FACE);                 			 // Режим при котором, тектуры накладываются только с одной стороны

	glewInit();
	glGenBuffers(1, &vbo);								// Получаем номер буфера
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);			// Делаем его активным
	glBufferData(GL_PIXEL_UNPACK_BUFFER, np * np * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);	// Задаем размер буфера
	cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);				// Регистрируем буфер для использования его памяти в CUDA
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);			// Деактивируем буфер

	glutMainLoop();
}
