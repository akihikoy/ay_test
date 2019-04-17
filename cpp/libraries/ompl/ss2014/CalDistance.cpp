#include "CalDistance.h"

using namespace std;


///normalize a 3d vector
int signR(double Z)
{
  int ret;
  if (Z > 0.0)    ret = 1;
  if (Z < 0.0)    ret = -1;
  if (Z == 0.0)   ret =0;

  return ret;
}

double CBRT(double Z)
{
  double ret;
  const double THIRD = 1./3.;
  //define cubic root as statement function
  //SIGN has different meanings in both C and Fortran
  // Was unable to use the sign command of C, so wrote my own
  // that why a new variable needs to be introduced that keeps track of the sign of
  // SIGN is supposed to return a 1, -1 or 0 depending on what the sign of the argument is
  ret = fabs(pow(fabs(Z),THIRD)) * signR(Z);
  return ret;
}

int cubic(double A[4], double X[3], int* L)
{
  const double PI = 3.1415926535897932;
  const double THIRD = 1./3.;
  double U[3],W, P, Q, DIS, PHI;
  int i;

  //define cubic root as statement function
  // In C, the function is defined outside of the cubic fct

  // ====determine the degree of the polynomial ====

  if (A[3] != 0.0)
  {
    //cubic problem
    W = A[2]/A[3]*THIRD;
    P = pow((A[1]/A[3]*THIRD - pow(W,2)),3);
    Q = -.5*(2.0*pow(W,3)-(A[1]*W-A[0])/A[3] );
    DIS = pow(Q,2)+P;
    if ( DIS < 0.0 )
    {
      //three real solutions!
      //Confine the argument of ACOS to the interval [-1;1]!
      PHI = acos(min(1.0,max(-1.0,Q/sqrt(-P))));
      P=2.0*pow((-P),(5.e-1*THIRD));
      for (i=0;i<3;i++)       U[i] = P*cos((PHI+2*((double)i)*PI)*THIRD)-W;
      X[0] = min(U[0], min(U[1], U[2]));
      X[1] = max(min(U[0], U[1]),max( min(U[0], U[2]), min(U[1], U[2])));
      X[2] = max(U[0], max(U[1], U[2]));
      *L = 3;
    }
    else
    {
      // only one real solution!
      DIS = sqrt(DIS);
      X[0] = CBRT(Q+DIS)+CBRT(Q-DIS)-W;
      *L=1;
    }
  }
  else if (A[2] != 0.0)
  {
    // quadratic problem
    P = 0.5*A[1]/A[2];
    DIS = pow(P,2)-A[0]/A[2];
    if (DIS > 0.0)
    {
      // 2 real solutions
      X[0] = -P - sqrt(DIS);
      X[1] = -P + sqrt(DIS);
      *L=2;
    }
    else
    {
      // no real solution
      *L=0;
    }
  }
  else if (A[1] != 0.0)
  {
    //linear equation
    X[0] =A[0]/A[1];
    *L=1;
  }
  else
  {
    //no equation
    *L=0;
  }
 /*
  *     ==== perform one step of a newton iteration in order to minimize
  *    round-off errors ====
  */
  for (i=0;i<*L;i++)
  {
    X[i] = X[i] - (A[0]+X[i]*(A[1]+X[i]*(A[2]+X[i]*A[3])))/(A[1]+X[i]*(2.0*A[2]+X[i]*3.0*A[3]));
  //      printf("\n X inside cubic %.15e\n", X[i]);
  }

  return 0;
}


int quartic(double dd[5], double sol[4], double soli[4], int* Nsol)
 {
  double AA[4], z[3];
  double a, b, c, d, f, p, q, r, zsol, xK2, xL, xK, sqp, sqm;
  int ncube, i;
  *Nsol = 0;

    for(i=0;i<4;++i)
    {
      sol[i] = 0.0;
    }

  if (dd[4] == 0.0)
  {
    printf("\n ERROR: NOT A QUARTIC EQUATION");
    return 0;
  }

  a = dd[4];
  b = dd[3];
  c = dd[2];
  d = dd[1];
  f = dd[0];

  p = (-3.0*pow(b,2) + 8.0 *a*c)/(8.0*pow(a,2));
  q = (pow(b,3) - 4.0*a*b*c + 8.0 *d*pow(a,2)) / (8.0*pow(a,3));
  r = (-3.0*pow(b,4) + 16.0 *a*pow(b,2)*c - 64.0 *pow(a,2)*b*d + 256.0 *pow(a,3)*f)/(256.0*pow(a,4));

  // Solve cubic resolvent
  AA[3] = 8.0;
  AA[2] = -4.0*p;
  AA[1] = -8.0*r;
  AA[0] = 4.0*p*r - pow(q,2);

  //printf("\n bcubic %.4e\t%.4e\t%.4e\t%.4e ", AA[0], AA[1], AA[2], AA[3]);
  cubic(AA, z, &ncube);
  //printf("\n acubic %.4e\t%.4e\t%.4e ", z[0], z[1], z[2]);

  zsol = - 1.e99;
  for (i=0;i<ncube;i++)   zsol = max(zsol, z[i]); //Not sure C has max fct
  z[0] =zsol;
  xK2 = 2.0*z[0] -p;
  xK = sqrt(xK2);
  xL = q/(2.0*xK);
  sqp = xK2 - 4.0 * (z[0] + xL);
  sqm = xK2 - 4.0 * (z[0] - xL);

  for (i=0;i<4;i++)       soli[i] = 0.0;
  if ( (sqp >= 0.0) && (sqm >= 0.0))
  {
    //*debug*/printf("\n case 1 \n");
    sol[0] = 0.5 * (xK + sqrt(sqp));
    sol[1] = 0.5 * (xK - sqrt(sqp));
    sol[2] = 0.5 * (-xK + sqrt(sqm));
    sol[3] = 0.5 * (-xK - sqrt(sqm));
    *Nsol = 4;
  }
  else if ( (sqp >= 0.0) && (sqm < 0.0))
  {
    //*debug*/printf("\n case 2 \n");
    sol[0] = 0.5 * (xK + sqrt(sqp));
    sol[1] = 0.5 * (xK - sqrt(sqp));
    sol[2] = -0.5 * xK;
    sol[3] = -0.5 * xK;
    soli[2] =  sqrt(-.25 * sqm);
    soli[3] = -sqrt(-.25 * sqm);
    *Nsol = 2;
  }
  else if ( (sqp < 0.0) && (sqm >= 0.0))
  {
    //*debug*/printf("\n case 3 \n");
    sol[0] = 0.5 * (-xK + sqrt(sqm));
    sol[1] = 0.5 * (-xK - sqrt(sqm));
    sol[2] = 0.5 * xK;
    sol[3] = 0.5 * xK;
    soli[2] =  sqrt(-0.25 * sqp);
    soli[3] = -sqrt(-0.25 * sqp);
    *Nsol = 2;
  }
  else if ( (sqp < 0.0) && (sqm < 0.0))
  {
    //*debug*/printf("\n case 4 \n");
    sol[0] = -0.5 * xK;
    sol[1] = -0.5 * xK;
    soli[0] =  sqrt(-0.25 * sqm);
    soli[1] = -sqrt(-0.25 * sqm);
    sol[2] = 0.5 * xK;
    sol[3] = 0.5 * xK;
    soli[2] =  sqrt(-0.25 * sqp);
    soli[3] = -sqrt(-0.25 * sqp);
    *Nsol = 0;
  }

  for (i=0;i<4;i++)
  {
     sol[i] -= b/(4.0*a);
  }
  return 0;
 }

inline vector3df CalDistance::getNormalizedVec(const vector3df& vec)
{
   float inv;
   float length = vec.getLength();
   if(length != 0.0) inv = (float)(1.0 / length);
   else inv = (float)0.0;
   vector3df result;
   result.X() = vec.X() * inv;
   result.Y() = vec.Y() * inv;
   result.Z() = vec.Z() * inv;
   return result;
}

/// point to line
std::vector<vector3df> CalDistance::GetClosestPointAndDist(const line3df& line, const vector3df& point, float &dist)
{
   std::vector<vector3df> res;
   res.push_back(point);
   float u = ( point.dotProduct(line.end-line.start) -
         line.start.dotProduct(line.end-line.start) )
         / line.getLengthSQ();
   vector3df P = line.start + u * (line.end-line.start);
   //*debug*/cout<<"u "<<u<<endl;
   if(line.isPointBetweenStartAndEnd(P))
   {
      //cout<<"Closest point to line is "<<P.X()<<" "<<P.Y()<<" "<<P.Z()<<endl;
      dist = line3df(point,P).getLength();
      res.push_back(P);
      return res;
   }
   else
   {
      if(line3df(point,line.start).getLength() < line3df(point,line.end).getLength())
      {
       //cout<<"Closest point to line is "<<line.start.X()<<" "<<line.start.Y()
       //    <<" "<<line.start.Z()<<endl;
       dist = line3df(point,line.start).getLength();
       res.push_back(line.start);
       return res;
      }
      else
      {
       //cout<<"Closest point to line is "<<line.end.X()<<" "<<line.end.Y()
        //   <<" "<<line.end.Z()<<endl;
       dist = line3df(point,line.end).getLength();
       res.push_back(line.end);
       return res;
      }
   }
}

/// line to line
std::vector<vector3df> CalDistance::GetClosestPointAndDist(const line3df& refline, const line3df& newline, float &dist)
{
   dist=-1; //dist unknown
   std::vector<vector3df> res;

   //for simplicity
   vector3df P1,P2,P3,P4;
   P1=refline.start;
   P2=refline.end;
   P3=newline.start;
   P4=newline.end;

   ///lines are parallel
   vector3df crossRes = (P2-P1).crossProduct(P4-P3);
// FIXME: eq
   if(crossRes.X() == 0.0 && crossRes.Y() == 0.0 && crossRes.Z() == 0.0) //lines are parallel
   {
      //*debug*/cout<<"lines are parallel"<<endl;
      float aux = -1.0, auxref = -1.0; //dist unknown
      std::vector<vector3df> auxVec, auxrefVec;

      auxVec = GetClosestPointAndDist(newline,P1,aux);
      auxref = aux;
      auxrefVec = auxVec;

      auxVec = GetClosestPointAndDist(newline,P2,aux);
      if(aux<auxref)
      {
   auxref = aux;
   auxrefVec = auxVec;
      }
      auxVec = GetClosestPointAndDist(refline,P3,aux);
      if(aux<auxref)
      {
   auxref = aux;
   auxrefVec = auxVec;
      }
      auxVec = GetClosestPointAndDist(refline,P4,aux);
      if(aux<auxref)
      {
   auxref = aux;
   auxrefVec = auxVec;
      }
      dist = auxref;
      return auxrefVec;
   }

   ///check if lines are on the same plane and intersect
   float ua, ub;
   ua=0.f; ub=0.f;
   vector3df vecA, vecB;
   bool uaFound = false;
   bool ubFound = false;

   //for ua
   vecA = (P2-P1).crossProduct(P4-P3);
   vecB = (P3-P1).crossProduct(P4-P3);

   // try to calculate ua from X
// FIXME: neq
   if(vecA.X() != 0.0)
   {
      if(vecB.X() != 0.0)
      {
       float ua_aux = vecB.X() / (float)vecA.X();
       float checkY = ua_aux * vecA.Y();
       if(checkY == vecB.Y())
       {
          float checkZ = ua_aux * vecA.Z();
          if(checkZ == vecB.Z())
          {
             uaFound = true;
             ua = ua_aux;
             //*debug*/cout<<"ua from X "<<ua<<endl;
          }
       }
      }
   }
   // try to calculate ua from Y
   if(vecA.Y() != 0.0 && !uaFound)
   {
      if(vecB.Y() != 0.0)
      {
   float ua_aux = vecB.Y() / (float)vecA.Y();
   float checkX = ua_aux * vecA.X();
   if(checkX == vecB.X())
   {
      float checkZ = ua_aux * vecA.Z();
      if(checkZ == vecB.Z())
      {
         uaFound = true;
         ua = ua_aux;
         //*debug*/cout<<"ua "<<ua<<endl;
      }
   }
      }
   }
   // try to calculate ua from Z
   if(vecA.Z() != 0.0 && !uaFound)
   {
      if(vecB.Z() != 0.0)
      {
   float ua_aux = vecB.Z() / (float)vecA.Z();
   float checkX = ua_aux * vecA.X();
   if(checkX == vecB.X())
   {
      float checkY = ua_aux * vecA.Y();
      if(checkY == vecB.Y())
      {
         uaFound = true;
         ua = ua_aux;
         //*debug*/cout<<"ua "<<ua<<endl;
      }
   }
      }
   }

   //for ub
   vecA = (P4-P3).crossProduct(P2-P1);
   vecB = (P1-P3).crossProduct(P2-P1);

   // try to calculate ub from X
   if(vecA.X() != 0.0)
   {
      if(vecB.X() != 0.0)
      {
   float ub_aux = vecB.X() / (float)vecA.X();
   float checkY = ub_aux * vecA.Y();
   if(checkY == vecB.Y())
   {
      float checkZ = ub_aux * vecA.Z();
      if(checkZ == vecB.Z())
      {
         ubFound = true;
         ub = ub_aux;
         //*debug*/cout<<"ub "<<ub<<endl;
      }
   }
      }
   }
   // try to calculate ub from Y
   if(vecA.Y() != 0.0 && !ubFound)
   {
      if(vecB.Y() != 0.0)
      {
   float ub_aux = vecB.Y() / (float)vecA.Y();
   float checkX = ub_aux * vecA.X();
   if(checkX == vecB.X())
   {
      float checkZ = ub_aux * vecA.Z();
      if(checkZ == vecB.Z())
      {
         ubFound = true;
         ub = ub_aux;
         //*debug*/cout<<"ub "<<ub<<endl;
      }
   }
      }
   }
   // try to calculate ub from Z
   if(vecA.Z() != 0.0 && !ubFound)
   {
      if(vecB.Z() != 0.0)
      {
   float ub_aux = vecB.Z() / (float)vecA.Z();
   float checkX = ub_aux * vecA.X();
   if(checkX == vecB.X())
   {
      float checkY = ub_aux * vecA.Y();
      if(checkY == vecB.Y())
      {
         ubFound = true;
         ub = ub_aux;
         //*debug*/cout<<"ub "<<ub<<endl;
      }
   }
      }
   }

   vector3df Pa_inter,Pb_inter; //intersecting points

   if(ua>0.0 && ua<1.0) //ua lies on ref line
   {
      if(ub>0.0 && ub<1.0)
      {
   //*debug*/cout<<"lines intersect"<<endl;
   Pa_inter = P1+ua*(P2-P1); //these should be the same point
   Pb_inter = P3+ub*(P4-P3); // "
   res.push_back(Pa_inter);
   res.push_back(Pb_inter);
   dist = 0.0;
   return res;
      }
   }

   /// lines in parallel planes
   // Pa and Pb lie on the lines
   float ta = 0.0; //ta = B/A
   float tb = 0.0; //tb = D/C

   float B = (P2-P1).dotProduct(P4-P3) * P1.dotProduct(P4-P3) -
       (P2-P1).dotProduct(P4-P3) * P3.dotProduct(P4-P3) -
       (P4-P3).getLengthSQ() * P1.dotProduct(P2-P1) +
       (P4-P3).getLengthSQ() * P3.dotProduct(P2-P1);

   float A = (P4-P3).getLengthSQ() * (P2-P1).getLengthSQ() -
       (P2-P1).dotProduct(P4-P3) * (P2-P1).dotProduct(P4-P3);

   float C = (P4-P3).getLengthSQ();

   if(A!=0.0 && C!=0.0)
   {
      ta=(float)B/(float)A;
   }
   else
   {
      cerr<<"Denominators A or C are zero"<<endl;
      //return res;
    vector<vector3df> tt;
    tt.push_back(vector3df(0,0,0));
    tt.push_back(vector3df(0,0,0));

    return tt;
   }

   float D = P1.dotProduct(P4-P3) + ta * (P2-P1).dotProduct(P4-P3) - P3.dotProduct(P4-P3);
   tb = (float)D/(float)C;

   //*debug*/cout << "ta "<<ta<<endl;
   //*debug*/cout << "tb "<<tb<<endl;

   vector3df Pa,Pb;
   Pa = P1 + ta * (P2-P1);
   Pb = P3 + tb * (P4-P3);

   if(ta>0.0 && ta<1.0)   //Pa lies on ref line
   {
      if(tb>0.0 && tb<1.0)      //Pb lies on new line
      {
   dist = (Pb-Pa).getLength();
   res.push_back(Pa);
   res.push_back(Pb);
   return res;
      }
      else          //Pb lies outside new line
      {
   if( (Pa-P3).getLength() < (Pa-P4).getLength() )  //compare Pa with P3 and P4
   {
      dist = (Pa-P3).getLength();
      res.push_back(Pa);
      res.push_back(P3);
      return res;
   }
   else
   {
      dist = (Pa-P4).getLength();
      res.push_back(Pa);
      res.push_back(P4);
      return res;
   }
      }
   }
   else       //Pa does not lie on ref line
   {
      if(tb>0.0 && tb<1.0)      //Pb lies on new line
      {
   if( (Pb-P1).getLength() < (Pb-P2).getLength() )
   {
      dist = (Pa-P1).getLength();
      res.push_back(Pb);
      res.push_back(P1);
      return res;
   }
   else
   {
      dist = (Pb-P2).getLength();
      res.push_back(Pb);
      res.push_back(P2);
      return res;
   }
      }
      else          //neither Pa nor Pb lie on the lines
      {
   float aux = 0.0;
   aux = (P1-P3).getLength();
   res.push_back(P1);
   res.push_back(P3);

   if( (P1-P4).getLength() < aux )
   {
      aux = (P1-P4).getLength();
      res[0] = P1;
      res[1] = P4;
   }
   if( (P2-P3).getLength() < aux )
   {
      aux = (P2-P3).getLength();
      res[0] = P2;
      res[1] = P3;
   }
   if( (P2-P4).getLength() < aux )
   {
      aux = (P2-P4).getLength();
      res[0] = P2;
      res[1] = P4;
   }
   dist = aux;
   return res;
      }
   }
   return res; //return empty vector if no solution
}

/// point to circle
std::vector<vector3df> CalDistance::GetClosestPointAndDist(const vector3df& P,//point
                const vector3df& C,//center
                const float &R, //radius
                const float& rotX, const float& rotY,
                const float& rotZ,float& dist) //to debug
{
   dist=-1; //dist unknown
   std::vector<vector3df> res;

   vector3df X1 = vector3df(C.X()+R*cos(0.f),C.Y()+R*sin(0.f),C.Z()); //any point in the circle
   X1.rotateYZBy(rotX,C); //rotation around X
   X1.rotateXZBy(rotY,C); //rotation around Y
   X1.rotateXYBy(rotZ,C); //rotation around Z

   vector3df X2 = vector3df(C.X()+R*cos(30.f),C.Y()+R*sin(30.f),C.Z()); //any point in the circle
   X2.rotateYZBy(rotX,C); //rotation around X
   X2.rotateXZBy(rotY,C); //rotation around Y
   X2.rotateXYBy(rotZ,C); //rotation around Z

   vector3df N = (C-X1).crossProduct(C-X2); //normal

   vector3df Q; //projection of P onto the circle plane
   Q = P - C - ( ((N.dotProduct(P-C))/N.getLengthSQ())*N );
   Q = P - ( N.dotProduct(P-C) * N/N.getLengthSQ() );
   //*debug*/cout<<"P "<<P.X()<<" "<<P.Y()<<" "<<P.Z()<<endl;
// FIXME: eq
   if(Q!=C)
   {
      vector3df X = (Q-C)/(float)(Q-C).getLength();
      X = C + R * X;
      //*debug*/cout<<"X "<<X.X()<<" "<<X.Y()<<" "<<X.Z()<<endl;
      res.push_back(P);
      res.push_back(X);
      dist = (P-X).getLength();
   }
   else //the projection is exactly over the center
   {
      //*debug*/cout<<"X1 "<<X1.X()<<" "<<X1.Y()<<" "<<X1.Z()<<endl;
      res.push_back(P);
      res.push_back(X1);
      dist = sqrt( R*R + (P-C).getLengthSQ() ); //hypotenuse
   }
   return res; //returns first the point and then the closest point on the circle
}

/// line to circle
std::vector<vector3df> CalDistance::GetClosestPointAndDist(const line3df& refline, //reference line
                const vector3df& C, //center
                const float &R, //radius
                const float& rotX, const float& rotY,
                const float& rotZ,float& dist) //to debug
{
   dist=-1; //dist unknown
   std::vector<vector3df> res;

   ///variables
   float a0,a1,b0,b1,c0,c1,c2; //substitutions for the first derivative
   float d4,d3,d2,d1,d0; //quadric coefficients
   vector3df P,B,M; //line is P=B+tM
   vector3df D; //D=B-C  (for simplification)
   vector3df N; //normal (unit vector)

   ///line P=B+tM
   M = (refline.end - refline.start);
   B = refline.start;
   D = B-C;  //for simplification

   ///calculate normal to the circle plane
   vector3df X1 = vector3df(C.X()+R*cos(0.f),C.Y()+R*sin(0.f),C.Z()); //any point in the circle
   X1.rotateYZBy(rotX,C); //rotation around X
   X1.rotateXZBy(rotY,C); //rotation around Y
   X1.rotateXYBy(rotZ,C); //rotation around Z
   //*debug*/MyDrawPoint(X1,device);
   vector3df X2 = vector3df(C.X()+R*cos(30.f),C.Y()+R*sin(30.f),C.Z()); //any point in the circle
   X2.rotateYZBy(rotX,C); //rotation around X
   X2.rotateXZBy(rotY,C); //rotation around Y
   X2.rotateXYBy(rotZ,C); //rotation around Z
   //*debug*/MyDrawPoint(X2,device);
   //N = (C-X1).crossProduct(C-X2); //normal
   N = (C-X2).crossProduct(C-X1); //normal
   //*debug*/cout<<"N "<<N.X()<<" "<<N.Y()<<" "<<N.Z()<<endl;
   N = getNormalizedVec(N); //make unit vec
   //*debug*/MyDrawLine(C,C+N,device);
   //*debug*/cout<<"N "<<N.X()<<" "<<N.Y()<<" "<<N.Z()<<endl;

   ///get coefficients to simplify first derivative
   a0 = M.dotProduct(D);
   a1 = M.dotProduct(M);
   b0 = M.dotProduct(D) - (N.dotProduct(M))*(N.dotProduct(D));
   b1 = M.dotProduct(M) - (N.dotProduct(M))*(N.dotProduct(M));
   c0 = D.dotProduct(D) - (N.dotProduct(D))*(N.dotProduct(D));
   c1 = b0;
   c2 = b1;

   ///coefficients of the quadric equation
   d0 = a0*a0*c0 - R*R*b0*b0;
   d1 = 2.f*a1*a0*c0 + 2.f*a0*a0*c1 - 2.f*R*R*b1*b0;
   d2 = a1*a1*c0 + 4.f*a1*a0*c1 + a0*a0*c2 - R*R*b1*b1;
   d3 = 2.f*a1*a1*c1 + 2.f*a1*a0*c2;
   d4 = a1*a1*c2;

   ///solve quartics with QuinticsSolver.hpp
   double dd[5]; //quartics coefficients
   double sol[4], soli[4]; //real and imaginary solutions
   int Nsol = 0; //number of real solutions
   for(int i=0;i<4;++i) //initializing is better ;)
   {
     sol[i] = 0.0;
     soli[i] = 0.0;
   }

   dd[0] = (double)d0; //independent term
   dd[1] = (double)d1;
   dd[2] = (double)d2;
   dd[3] = (double)d3;
   dd[4] = (double)d4;
   quartic(dd, sol, soli, &Nsol);
   vector<float> solVec;
   for(int i=0;i<4;++i)
   {
     //*debug*/cout<<"sol["<<i<<"] "<<sol[i]<<endl;
      if(!isinf(sol[i]))
    {
   solVec.push_back((float)sol[i]);
    }
   }
   if(solVec.size()==0)
   {
      cerr<<"No solutions found for the quartics"<<endl;
      return res;
   }
   //*debug*/cout<<"Real solutions: "<<solVec.size()<<endl;

   P = B + solVec[0]*M; //one point in the line that satifies the quartics

   ///squared distance
   vector3df Q_C = P - C - (N.dotProduct(P-C))*N;
   vector3df W = getNormalizedVec(Q_C);
   float sqDist = R*R + (C-P).getLengthSQ() + 2.f*R*(C-P).dotProduct(W);
   //*debug*/cout<<"\nsqDist "<<sqDist<<endl;
   int solIndex = 0;

   /*debug*/cout<<endl;
   for(size_t i=0;i<solVec.size();++i)
   {
      P = B + solVec[i]*M;
      vector3df _Q_C = P - C - (N.dotProduct(P-C))*N;
      vector3df _W = getNormalizedVec(_Q_C);
      float _sqDist = R*R + (C-P).getLengthSQ() + 2.f*R*(C-P).dotProduct(_W);
      //*debug*/cout<<setprecision(10)<<"_sqDist ("<<i<<") = "<<_sqDist<<endl;
      if(_sqDist < sqDist)
      {
   sqDist = _sqDist;
   solIndex = i;
      }
   }
   //*debug*/cout<<"solIndex "<<solIndex<<endl;
   P = B + solVec[solIndex]*M;

   /// when the closest point lies out of the line
   if(solVec[solIndex]<0.0 || solVec[solIndex]>1.0)
   {
      P = refline.start;
      vector3df _Q_C = P - C - (N.dotProduct(P-C))*N;
      vector3df _W = getNormalizedVec(_Q_C);
      float sqDist_start = R*R + (C-P).getLengthSQ() + 2.f*R*(C-P).dotProduct(_W);

      P = refline.end;
      _Q_C = P - C - (N.dotProduct(P-C))*N;
      _W = getNormalizedVec(_Q_C);
      float sqDist_end = R*R + (C-P).getLengthSQ() + 2.f*R*(C-P).dotProduct(_W);

      if(sqDist_start <= sqDist_end)
   P = refline.start;
      else
   P = refline.end;
   }
   //*debug*/MyDrawPoint(P,device);

   ///solve point to circle
   res = GetClosestPointAndDist(P,C,R,rotX,rotY,rotZ,dist);

   return res;
}


/// circle to circle
std::vector<vector3df> CalDistance::GetClosestPointAndDist(const vector3df& C1, //reference circle
                const vector3df& C2, //center
                const float& R1, //radius ref circle
                const float& R2, //radius
                const float& rotX1, const float& rotX2,
                const float& rotY1, const float& rotY2,
                const float& rotZ1, const float& rotZ2,
                float& dist) //to debug
{
   dist=-1.0; //dist unknown
   std::vector<vector3df> res;

   /// NOTE: circle ref is Cir1, circle is Cir2

   float distC1C2 = -1.0;

   float distC1Cir2 = -1.0;
   float distCir2Cir1 = -1.0;

   float distC2Cir1 = -1.0;
   float distCir1Cir2 = -1.0;

   //*debug*/PrintVec3d(C1,"C1");
   //*debug*/PrintVec3d(C2,"C2");
   //*debug*/cout<<"R1 "<<R1<<endl;
   //*debug*/cout<<"R2 "<<R2<<endl;

   ///store results
   std::vector<std::vector<vector3df> > pointsVec;
   std::vector<float> distsVec;

   ///distance between centers
   distC1C2=(C2-C1).getLength();
   std::vector<vector3df> resC1C2;
   resC1C2.push_back(C1);
   resC1C2.push_back(C2);

   ///distance from center 1 to the circle 2
   std::vector<vector3df> resC1Cir2 = GetClosestPointAndDist(C1,C2,R2,
                   rotX2,rotY2,rotZ2,
                   distC1Cir2);
   //*debug*/cout<<setprecision(10)<<"distC1Cir2 "<<distC1Cir2<<endl;
   //*debug*/PrintVecOfVec3d(resC1Cir2,"resC1Cir2");

   ///distance from the closest point (from the previous result) in circle 2 to circle 1
   std::vector<vector3df> resCir2Cir1 = GetClosestPointAndDist(resC1Cir2[1],C1,R1,
                   rotX1,rotY1,rotZ1,
                   distCir2Cir1);
   //*debug*/cout<<setprecision(10)<<"distCir2Cir1 "<<distCir2Cir1<<endl;
   //*debug*/PrintVecOfVec3d(resCir2Cir1,"resCir2Cir1");
   pointsVec.push_back(resCir2Cir1);
   distsVec.push_back(distCir2Cir1);

   ///distance from center 2 to the circle 1
   std::vector<vector3df> resC2Cir1 = GetClosestPointAndDist(C2,C1,R1,
                   rotX1,rotY1,rotZ1,
                   distC2Cir1);
   //*debug*/cout<<setprecision(10)<<"distC2Cir1 "<<distC2Cir1<<endl;
   //*debug*/PrintVecOfVec3d(resC2Cir1,"resC2Cir1");

   ///distance from the closest point (from the previous result) in circle 1 to circle 2
   std::vector<vector3df> resCir1Cir2 = GetClosestPointAndDist(resC2Cir1[1],C2,R2,
                   rotX2,rotY2,rotZ2,
                   distCir1Cir2);
   //*debug*/cout<<setprecision(10)<<"distCir1Cir2 "<<distCir1Cir2<<endl;
   //*debug*/PrintVecOfVec3d(resCir1Cir2,"resCir1Cir2");
   pointsVec.push_back(resCir1Cir2);
   distsVec.push_back(distCir1Cir2);

   float auxdist = distC1C2; ///TODO check
   int auxIndex = 0;

   for(size_t i=0;i<pointsVec.size() && i<distsVec.size();++i)
   {
      if(distsVec[i] < auxdist)
      {
   auxdist = distsVec[i];
   auxIndex = i;
      }
   }

   res = pointsVec[auxIndex];
   dist = distsVec[auxIndex];

   ///TODO: this method is inexact, try with the 8th degree polynomial

   return res;
}

/// cylinder 2 cylinder
std::vector<vector3df> CalDistance::GetClosestPointAndDist(const vector3df& C1_1, //begin cylinder
                        const vector3df& C1_2, //
                const vector3df& C2_1, //begin cylinder
                        const vector3df& C2_2, //
                const float& R1, //radius ref circle
                const float& R2, //radius
                const float& rotX1, const float& rotX2,
                const float& rotY1, const float& rotY2,
                const float& rotZ1, const float& rotZ2,
                float& dist)
{
  //part 1
  std::vector<vector3df> Pvec;
  line3df l1,l2;
  l1 = line3df(C1_1,C1_2);
  l2 = line3df(C2_1,C2_2);
  Pvec = GetClosestPointAndDist(l1,l2,dist); //line to line

  //part 2
  if(Pvec.size() > 0)
  Pvec = GetClosestPointAndDist(Pvec[0],Pvec[1],
           R1,R2,
           rotX1,rotX2,
           rotY1,rotY2,
           rotZ1,rotZ2,
           dist); //circle 2 circle
  return Pvec;
}

/// sphere to cylinder
std::vector<vector3df> CalDistance::GetClosestPointAndDist(const vector3df& C_1, //begin cylinder
                        const vector3df& C_2, //
                const vector3df& center, //sphere center
                const float& Rcyl, //radius cylinder
                const float& Rsph, //radius sphere
                const float& rotX1,
                const float& rotY1,
                const float& rotZ1,
                float& dist)
{
  std::vector<vector3df> Pvec;

  line3df cylAxis(C_1,C_2);
  Pvec = GetClosestPointAndDist(cylAxis, center, dist); // point to line (returns the point as first element)

  if(Pvec.size() == 2)
  {
    if(Pvec[1]==C_1)
    {
      // point to circle
      Pvec = GetClosestPointAndDist(center,//point
                      C_1,//center
                      Rcyl, //radius
            rotX1, rotY1, rotZ1,
                      dist);
      if(Pvec.size() == 2)
        dist -= Rsph; //minus the sphere radius
      if(dist < 0.0)
        dist = 0.0; ///TODO check
    }
    else if(Pvec[1]==C_2)
    {
      // point to circle
      Pvec = GetClosestPointAndDist(center,//point
                      C_2,//center
                      Rcyl, //radius
            rotX1, rotY1, rotZ1,
                      dist);
      if(Pvec.size() == 2)
        dist -= Rsph; //minus the sphere radius
      if(dist < 0.0)
        dist = 0.0; ///TODO check
    }
    else
    {
      dist -= (Rsph+Rcyl); //minus the sphere and cylynder's radii
      if(dist < 0.0)
        dist = 0.0; ///TODO check
    }
  }

  return Pvec;
}

/// sphere to cylinder extremes (like sphere to sphere), for debugging //not efficient due to double calculation
std::vector<vector3df> CalDistance::GetClosestPointAndDist(const vector3df& H_C1, //begin cylinder
                                const vector3df& H_C2, //
                                const vector3df& R_C1, //begin cylinder
                                const vector3df& R_C2, //
                                const float& H_radius, //radius of sphere
                                const float& Rr_radius,
                                float& dist)
{
  std::vector<vector3df> auxPvec, Pvec;
  float auxDist, minDist;

  minDist = (H_C1-R_C1).getLength();
  Pvec.push_back(H_C1); Pvec.push_back(R_C1);
  auxDist = (H_C1-R_C2).getLength();
  if(auxDist<minDist)
  {
    minDist = auxDist;
    auxPvec.push_back(H_C1); auxPvec.push_back(R_C2);
    Pvec = auxPvec;
  }
  auxDist = (H_C2-R_C1).getLength();
  if(auxDist<minDist)
  {
    minDist = auxDist;
    auxPvec.clear();
    auxPvec.push_back(H_C2); auxPvec.push_back(R_C1);
    Pvec = auxPvec;
  }
  auxDist = (H_C2-R_C2).getLength();
  if(auxDist<minDist)
  {
    minDist = auxDist;
    auxPvec.clear();
    auxPvec.push_back(H_C2); auxPvec.push_back(R_C2);
    Pvec = auxPvec;
  }

  dist = minDist;
  return Pvec;
}

std::vector<vector3df> CalDistance::GetClosestPointAndDist(const vector3df& C_1, //begin cylinder
                                     const vector3df& C_2, //
                                     const vector3df& center, //sphere center
                                     const float& cyl_R, //radius cylinder
                                     const float& sph_R, //radius sphere
                                     float& dist)
{
  std::vector<vector3df> Pvec;
  float minDist, auxDist;

  minDist = (C_1-center).getLength();
  auxDist = (C_2-center).getLength();



  if(auxDist < minDist)
  {
    minDist = auxDist;
    Pvec.push_back(C_2);
  }
  else
    Pvec.push_back(C_1);

  Pvec.push_back(center);

  dist = minDist;
  return Pvec;
}