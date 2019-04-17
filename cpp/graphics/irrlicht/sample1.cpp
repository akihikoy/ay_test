// src from http://lesson.ifdef.jp/005.html
#include <irrlicht/irrlicht.h>

using namespace irr;
using namespace core;
using namespace video;
using namespace scene;

f32 rx = 0;
f32 ry = 0;

void makeScene(IVideoDriver *driver)
{
  matrix4 mat;

  //属性設定
  SMaterial Material;
  Material.Lighting = false;                      //ライト設定
//      Material.BackfaceCulling = false;       //カリング設定
  driver->setMaterial(Material);

  //四面体作成
  u16 list[] = {0,1,2, 2,1,3, 1,0,3, 0,2,3};
  S3DVertex ver[4];
  ver[0] = S3DVertex( 0, 1,0,  0,0,0, 0xFFFF0000, 0,0);//上　：赤
  ver[1] = S3DVertex( 1,-1,0,  0,0,0, 0xFF00FF00, 0,0);//右下：緑
  ver[2] = S3DVertex(-1,-1,0,  0,0,0, 0xFF0000FF, 0,0);//左下：青
  ver[3] = S3DVertex( 0, 0,2,  0,0,0, 0xFFFF00FF, 0,0);//奥　：紫
  mat.setTranslation(vector3df(-2,0,0));//左に2移動
  mat.setRotationRadians(vector3df(rx,0,0));//X軸で回転
  driver->setTransform (ETS_WORLD, mat);//ワールドに反映
  driver->drawIndexedTriangleList(&ver[0], 4, &list[0], 4);

  //六面体作成
  u16 list1[] = { 0, 1, 2,  2, 1, 3};
  u16 list2[] = { 4, 5, 6,  6, 5, 7};
  u16 list3[] = { 8, 9,10, 10, 9,11};
  u16 list4[] = {12,13,14, 14,13,15};
  u16 list5[] = {16,17,18, 18,17,19};
  u16 list6[] = {20,21,22, 22,21,23};
  S3DVertex ver2[24];
  ver2[ 0] = S3DVertex(-1, 1,-1,  0,0,0, 0xFFFF0000, 0,0);//前左上
  ver2[ 1] = S3DVertex( 1, 1,-1,  0,0,0, 0xFFFF0000, 0,0);//前右上
  ver2[ 2] = S3DVertex(-1,-1,-1,  0,0,0, 0xFFFF0000, 0,0);//前左下
  ver2[ 3] = S3DVertex( 1,-1,-1,  0,0,0, 0xFFFF0000, 0,0);//前右下

  ver2[ 4] = S3DVertex( 1, 1, 1,  0,0,0, 0xFF00FF00, 0,0);//奥左上
  ver2[ 5] = S3DVertex(-1, 1, 1,  0,0,0, 0xFF00FF00, 0,0);//奥右上
  ver2[ 6] = S3DVertex( 1,-1, 1,  0,0,0, 0xFF00FF00, 0,0);//奥左下
  ver2[ 7] = S3DVertex(-1,-1, 1,  0,0,0, 0xFF00FF00, 0,0);//奥右下

  ver2[ 8] = S3DVertex(-1, 1, 1,  0,0,0, 0xFF0000FF, 0,0);//左左上
  ver2[ 9] = S3DVertex(-1, 1,-1,  0,0,0, 0xFF0000FF, 0,0);//左右上
  ver2[10] = S3DVertex(-1,-1, 1,  0,0,0, 0xFF0000FF, 0,0);//左左下
  ver2[11] = S3DVertex(-1,-1,-1,  0,0,0, 0xFF0000FF, 0,0);//左右下

  ver2[12] = S3DVertex( 1, 1,-1,  0,0,0, 0xFFFFFF00, 0,0);//右左上
  ver2[13] = S3DVertex( 1, 1, 1,  0,0,0, 0xFFFFFF00, 0,0);//右右上
  ver2[14] = S3DVertex( 1,-1,-1,  0,0,0, 0xFFFFFF00, 0,0);//右左下
  ver2[15] = S3DVertex( 1,-1, 1,  0,0,0, 0xFFFFFF00, 0,0);//右右下

  ver2[16] = S3DVertex( 1, 1,-1,  0,0,0, 0xFFFF00FF, 0,0);//上左上
  ver2[17] = S3DVertex(-1, 1,-1,  0,0,0, 0xFFFF00FF, 0,0);//上右上
  ver2[18] = S3DVertex( 1, 1, 1,  0,0,0, 0xFFFF00FF, 0,0);//上左下
  ver2[19] = S3DVertex(-1, 1, 1,  0,0,0, 0xFFFF00FF, 0,0);//上右下

  ver2[20] = S3DVertex(-1,-1,-1,  0,0,0, 0xFF00FFFF, 0,0);//下左上
  ver2[21] = S3DVertex( 1,-1,-1,  0,0,0, 0xFF00FFFF, 0,0);//下右上
  ver2[22] = S3DVertex(-1,-1, 1,  0,0,0, 0xFF00FFFF, 0,0);//下左下
  ver2[23] = S3DVertex( 1,-1, 1,  0,0,0, 0xFF00FFFF, 0,0);//下右下

  mat.setTranslation(vector3df(2,0,0));//右に2移動
  mat.setRotationRadians(vector3df(rx,ry,0));//Y軸で回転
  driver->setTransform (ETS_WORLD, mat);//ワールドに反映
  driver->drawIndexedTriangleList(&ver2[0], 24, &list1[0], 2);
  driver->drawIndexedTriangleList(&ver2[0], 24, &list2[0], 2);
  driver->drawIndexedTriangleList(&ver2[0], 24, &list3[0], 2);
  driver->drawIndexedTriangleList(&ver2[0], 24, &list4[0], 2);
  driver->drawIndexedTriangleList(&ver2[0], 24, &list5[0], 2);
  driver->drawIndexedTriangleList(&ver2[0], 24, &list6[0], 2);

  rx += 0.002f;
  ry += 0.001f;
}

int main()
{
  IrrlichtDevice *device = createDevice(video::EDT_SOFTWARE,dimension2d<u32>(320,240),16,false,false,false);
  IVideoDriver *driver = device->getVideoDriver();
  ISceneManager* smgr  = device->getSceneManager();

  device->setWindowCaption(L"Irrlicht");//ウインドウタイトル設定

  //カメラ設定
  smgr->addCameraSceneNode(0, vector3df(0,0,-5), vector3df(0,0,0));

  while(device->run())
  {
    driver->beginScene(true,true,0xFF6060FF);

    //図形作成
    makeScene(driver);

    //シーンの描画
    smgr->drawAll();

    driver->endScene();
  }

  driver->drop();

  return 0;
}

