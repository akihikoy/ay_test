// src: http://hael.jugem.jp/?eid=234
#include <GL/freeglut.h>

int main(int argc, char* argv[])
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

  glutCreateWindow("win0");
  int win0 = glutGetWindow();
  glutShowWindow();

  while(true)
  {
    glutMainLoopEvent();

    glutSetWindow(win0);
    glClearColor(1.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3d(0.0, 0.0, 1.0);
    glBegin(GL_LINE_LOOP);
    glVertex2d(-0.9, -0.9);
    glVertex2d(0.9, -0.9);
    glVertex2d(0.9, 0.9);
    glVertex2d(-0.9, 0.9);
    glEnd();

    glColor3d(0.0, 0.5, 0.5);
    glBegin(GL_POLYGON);
    glVertex2d(-0.5, -0.5);
    glVertex2d(0.5, -0.5);
    glVertex2d(0.5, 0.5);
    glVertex2d(-0.5, 0.5);
    glEnd();

    glColor3d(0.0, 1.0, 0.0);
    glBegin(GL_POINTS);
    glVertex2d(0.0, 0.0);
    glEnd();

    glFlush();
    glutSwapBuffers();
  }

  return 0;
}

