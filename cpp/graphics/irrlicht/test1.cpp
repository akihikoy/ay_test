// src from http://lesson.ifdef.jp/005.html
#include <irrlicht/irrlicht.h>
#include <iostream>

using namespace irr;
using namespace core;
using namespace video;
using namespace scene;

f32 rx = 0;
f32 ry = 0;

void makeScene(IrrlichtDevice *device)
{
IVideoDriver *driver = device->getVideoDriver();
  matrix4 mat;

  {
    //属性設定
    SMaterial Material;
    // Material.Lighting = false;                      //ライト設定
    Material.Lighting = true;                      //ライト設定
  //      Material.BackfaceCulling = false;       //カリング設定
    driver->setMaterial(Material);

    //四面体作成
    u16 list[] = {0,1,2, 2,1,3, 1,0,3, 0,2,3};
    S3DVertex ver[4];
    ver[0] = S3DVertex( 0, 1,0,  0,0,1, 0xFFFF0000, 0,0);//上　：赤
    ver[1] = S3DVertex( 1,-1,0,  0,0,-1, 0xFF00FF00, 0,0);//右下：緑
    ver[2] = S3DVertex(-1,-1,0,  0,0,-1, 0xFF0000FF, 0,0);//左下：青
    ver[3] = S3DVertex( 0, 0,2,  0,0,-1, 0xFFFF00FF, 0,0);//奥　：紫
    mat.setTranslation(vector3df(-2,0,0));//左に2移動
    mat.setRotationRadians(vector3df(rx,0,0));//X軸で回転
    driver->setTransform (ETS_WORLD, mat);//ワールドに反映
    driver->drawIndexedTriangleList(&ver[0], 4, &list[0], 4);
  }

{
// scene::ISceneManager* sm = device->getSceneManager();
// const scene::IGeometryCreator* geom = sm->getGeometryCreator();
// scene::IMesh* mesh;
// scene::ISceneNode* node;
// mesh = geom->createSphereMesh(5.0f, 32, 32);
// node = sm->addMeshSceneNode(mesh, NULL, -1, core::vector3df(0, 0, 50));
// node->getMaterial(0).EmissiveColor = video::SColor(255, 30, 30, 30);
// mesh->drop();

// this may be a common object in the whole program
scene::IAnimatedMesh* sphereMesh = device->getSceneManager()->addSphereMesh("atom", 1, 32, 32);

mat.setTranslation(vector3df(+2,ry,0));
mat.setRotationRadians(vector3df(ry,0,0));
driver->setTransform (ETS_WORLD, mat);

video::SMaterial material;
material.Lighting=true;
material.MaterialType=video::EMT_SOLID;
material.ColorMaterial=video::ECM_DIFFUSE;
// material.ColorMaterial=video::ECM_NONE;
material.AmbientColor.set(255, 80, 80, 80);
material.DiffuseColor.set(255, 120, 30, 210);
material.SpecularColor.set(255,80,80,80);
// material.AmbientColor.set(255, 10, 0, 0);
// material.DiffuseColor.set(255, 0, 10, 0);
// material.SpecularColor.set(255,0, 0, 10);
material.Shininess = 8.f;
driver->setMaterial(material);

driver->drawMeshBuffer (sphereMesh->getMeshBuffer(0));
}

// std::cerr<<rx<<"  "<<ry<<std::endl;
  rx += 0.002f;
  ry += 0.001f;
}

int main()
{
  // IrrlichtDevice *device = createDevice(video::EDT_SOFTWARE,dimension2d<u32>(320,240),16,false,false,false);
  IrrlichtDevice *device = createDevice(video::EDT_BURNINGSVIDEO,dimension2d<u32>(320,240),16,false,false,false);
  // IrrlichtDevice *device = createDevice(video::EDT_OPENGL,dimension2d<u32>(320,240),16,false,false,false);
  IVideoDriver *driver = device->getVideoDriver();
  ISceneManager* smgr  = device->getSceneManager();

  device->setWindowCaption(L"Irrlicht");//ウインドウタイトル設定

  //カメラ設定
  scene::ICameraSceneNode* cam = smgr->addCameraSceneNode(0, vector3df(0,0,-5), vector3df(0,0,0));
smgr->setActiveCamera(cam);

// //ライト作成
// ILightSceneNode *light = smgr->addLightSceneNode(
        // 0, vector3df(0,0,0), SColorf(0xFFFFFFFF), 1.0f);
// //ライトタイプ
// light->getLightData().Type = ELT_DIRECTIONAL;
// //light->getLightData().Type = ELT_POINT;
// //light->getLightData().Type = ELT_SPOT;
// //減衰値設定
// light->getLightData().Attenuation = vector3df(0.05f, 0.05f, 0.05f);

smgr->addLightSceneNode(0, core::vector3df(5,5,5),SColorf(0xFFFFFFFF));
smgr->setAmbientLight(video::SColorf(0.3f,0.3f,0.3f));

  while(device->run())
  {
    driver->beginScene(true,true,0xFF6060FF);
    // driver->beginScene(true,true,0xFF000000);

    //図形作成
    makeScene(device);

    //シーンの描画
    smgr->drawAll();

    driver->endScene();
  }

  driver->drop();

  return 0;
}

