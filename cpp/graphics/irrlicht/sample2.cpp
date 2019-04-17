// src: http://lesson.ifdef.jp/007.html
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
        f32 a = 1.0f;
        f32 b = 0.125f;
        f32 c = 0.25f;
        f32 d = 0.375f;
        f32 e = 0.5f;
        f32 f = 0.625f;
        f32 g = 0.75f;

        //テクスチャ読み込み
        ITexture* Texture;
        Texture = driver->getTexture("data.png");
        SMaterial Material;
//      Material.Lighting = false;
        //Material.Textures[0] = Texture;//1.3
        Material.TextureLayer[0].Texture = Texture;//1.4.1
        driver->setMaterial(Material);

        //六面体作成
        u16 list1[] = { 0, 1, 2,  2, 1, 3};
        u16 list2[] = { 4, 5, 6,  6, 5, 7};
        u16 list3[] = { 8, 9,10, 10, 9,11};
        u16 list4[] = {12,13,14, 14,13,15};
        u16 list5[] = {16,17,18, 18,17,19};
        u16 list6[] = {20,21,22, 22,21,23};

        S3DVertex ver2[24];
        ver2[ 0] = S3DVertex(-1, 1,-1,  0,0,-1, 0xFFFFFFFF, f,0);//前左上6
        ver2[ 1] = S3DVertex( 1, 1,-1,  0,0,-1, 0xFFFFFFFF, g,0);//前右上
        ver2[ 2] = S3DVertex(-1,-1,-1,  0,0,-1, 0xFFFFFFFF, f,a);//前左下
        ver2[ 3] = S3DVertex( 1,-1,-1,  0,0,-1, 0xFFFFFFFF, g,a);//前右下

        ver2[ 4] = S3DVertex( 1, 1, 1,  0,0,1,  0xFFFFFFFF, 0,0);//奥左上1
        ver2[ 5] = S3DVertex(-1, 1, 1,  0,0,1,  0xFFFFFFFF, b,0);//奥右上
        ver2[ 6] = S3DVertex( 1,-1, 1,  0,0,1,  0xFFFFFFFF, 0,a);//奥左下
        ver2[ 7] = S3DVertex(-1,-1, 1,  0,0,1,  0xFFFFFFFF, b,a);//奥右下

        ver2[ 8] = S3DVertex(-1, 1, 1,  -1,0,0, 0xFFFFFFFF, b,0);//左左上2
        ver2[ 9] = S3DVertex(-1, 1,-1,  -1,0,0, 0xFFFFFFFF, c,0);//左右上
        ver2[10] = S3DVertex(-1,-1, 1,  -1,0,0, 0xFFFFFFFF, b,a);//左左下
        ver2[11] = S3DVertex(-1,-1,-1,  -1,0,0, 0xFFFFFFFF, c,a);//左右下

        ver2[12] = S3DVertex( 1, 1,-1,  1,0,0,  0xFFFFFFFF, e,0);//右左上5
        ver2[13] = S3DVertex( 1, 1, 1,  1,0,0,  0xFFFFFFFF, f,0);//右右上
        ver2[14] = S3DVertex( 1,-1,-1,  1,0,0,  0xFFFFFFFF, e,a);//右左下
        ver2[15] = S3DVertex( 1,-1, 1,  1,0,0,  0xFFFFFFFF, f,a);//右右下

        ver2[16] = S3DVertex( 1, 1,-1,  0,1,0,  0xFFFFFFFF, d,0);//上左上3
        ver2[17] = S3DVertex(-1, 1,-1,  0,1,0,  0xFFFFFFFF, c,0);//上右上
        ver2[18] = S3DVertex( 1, 1, 1,  0,1,0,  0xFFFFFFFF, d,a);//上左下
        ver2[19] = S3DVertex(-1, 1, 1,  0,1,0,  0xFFFFFFFF, c,a);//上右下

        ver2[20] = S3DVertex(-1,-1,-1,  0,-1,0, 0xFFFFFFFF, d,0);//下左上4
        ver2[21] = S3DVertex( 1,-1,-1,  0,-1,0, 0xFFFFFFFF, e,0);//下右上
        ver2[22] = S3DVertex(-1,-1, 1,  0,-1,0, 0xFFFFFFFF, d,a);//下左下
        ver2[23] = S3DVertex( 1,-1, 1,  0,-1,0, 0xFFFFFFFF, e,a);//下右下

        mat.setRotationRadians(vector3df(rx,ry,0));//Y軸で回転
        driver->setTransform (ETS_WORLD, mat);//ワールドに反映
        driver->drawIndexedTriangleList(&ver2[0], 24, &list1[0], 2);
        driver->drawIndexedTriangleList(&ver2[0], 24, &list2[0], 2);
        driver->drawIndexedTriangleList(&ver2[0], 24, &list3[0], 2);
        driver->drawIndexedTriangleList(&ver2[0], 24, &list4[0], 2);
        driver->drawIndexedTriangleList(&ver2[0], 24, &list5[0], 2);
        driver->drawIndexedTriangleList(&ver2[0], 24, &list6[0], 2);

        rx += 0.001f;
        ry += 0.001f;
}

int main()
{
        IrrlichtDevice *device = createDevice(video::EDT_BURNINGSVIDEO,dimension2d<u32>(320,240),16,false,false,false);
        IVideoDriver *driver = device->getVideoDriver();
        ISceneManager* smgr  = device->getSceneManager();

        device->setWindowCaption(L"Irrlicht");//ウインドウタイトル設定

        //ライト
        smgr->addLightSceneNode(0, vector3df(0,0,-10), SColorf(0xFFFFFFFF), 10.0f);

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

