// src: http://doc.coin3d.org/SoQt/

#include <Inventor/Qt/SoQt.h>
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoCone.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTransform.h>

#include <iostream>
#include <unistd.h>
#include <boost/thread.hpp>
// #include <boost/bind.hpp>

SoTransform *xf1(NULL);
double y(0.0);

void loop()
{
  while(y<5.0)
  {
    xf1->translation.setValue(0.0, y, 0.0);
    y+= 0.01;
    std::cout<<y<<std::endl;
    usleep(10000);
  }
}

int main(int argc, char ** argv)
{
  // Initializes SoQt library (and implicitly also the Coin and Qt
  // libraries). Returns a top-level / shell Qt window to use.
  QWidget * mainwin = SoQt::init(argc, argv, argv[0]);

  // Make a dead simple scene graph by using the Coin library, only
  // containing a single yellow cone under the scenegraph root.
  SoSeparator * root = new SoSeparator;
  root->ref();

  SoBaseColor * col = new SoBaseColor;
  col->rgb = SbColor(1, 1, 0);
  root->addChild(col);

  root->addChild(new SoCone);

  xf1 = new SoTransform;
  xf1->translation.setValue(0.0, 3.0, 0.0);
  root->addChild(xf1);
  root->addChild(new SoCone);

  // Use one of the convenient SoQt viewer classes.
  SoQtExaminerViewer * eviewer = new SoQtExaminerViewer(mainwin);
  eviewer->setSceneGraph(root);
  eviewer->show();

  boost::thread th(&loop);

  // Pop up the main window.
  SoQt::show(mainwin);
  // Loop until exit.
  SoQt::mainLoop();

  th.join();

  // Clean up resources.
  delete eviewer;
  root->unref();
  SoQt::done();

  return 0;
}
