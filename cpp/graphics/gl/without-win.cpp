// src: http://hael.jugem.jp/?eid=234
#include <GL/freeglut.h>

int main(int argc, char* argv[])
{
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

        glutCreateWindow("win0");
        int win0 = glutGetWindow();
        glutShowWindow();

        glutCreateWindow("win1");
        int win1 = glutGetWindow();
        glutShowWindow();

        // glutDisplayFunc(display);
        // glutMainLoop();

        while(true)
        {
                glutMainLoopEvent();

                glutSetWindow(win0);
                glClearColor(1.f, 0.f, 0.f, 1.f);
                glClear(GL_COLOR_BUFFER_BIT);
                glFlush();
                glutSwapBuffers();

                glutSetWindow(win1);
                glClearColor(0.f, 0.f, 1.f, 1.f);
                glClear(GL_COLOR_BUFFER_BIT);
                glFlush();
                glutSwapBuffers();
        }

        return 0;
}

