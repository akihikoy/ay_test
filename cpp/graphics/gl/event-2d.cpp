// src: http://hael.jugem.jp/?eid=234
#include <GL/freeglut.h>
#include <iostream>
using namespace std;

void resize(int w, int h)
{
  glViewport(0, 0, w, h);
  glLoadIdentity();
  glOrtho(-w / 200.0, w / 200.0, -h / 200.0, h / 200.0, -1.0, 1.0);
}

void mouse(int button, int state, int x, int y)
{
  switch(button)
  {
  case GLUT_LEFT_BUTTON:    cout<<"left";   break;
  case GLUT_MIDDLE_BUTTON:  cout<<"middle"; break;
  case GLUT_RIGHT_BUTTON:   cout<<"right";  break;
  default:  cout<<"?("<<button<<")";  break;
  }
  cout<<" mouse is ";
  switch(state)
  {
  case GLUT_UP:    cout<<"up";   break;
  case GLUT_DOWN:  cout<<"down"; break;
  default:  break;
  }

  cout<<" at "<<x<<", "<<y<<endl;
}

void motion(int x, int y)
{
  cout<<" at "<<x<<", "<<y<<endl;
}

bool EXECUTING(true);

void keyboard(unsigned char key, int x, int y)
{
  switch(key)
  {
  case '\033': EXECUTING=false; break;
  default: break;
  }
}

int main(int argc, char* argv[])
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

  glutInitWindowSize(400, 300);
  glutCreateWindow("win0");
  int win0 = glutGetWindow();
  glutShowWindow();

  glutReshapeFunc(resize);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);

  while(EXECUTING)
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

