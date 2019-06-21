#include <math.h>   
#include <stdlib.h> 
#include <stdio.h>  

#include <gl\glew.h>
#include <gl\freeglut.h>
#include <thread>
#include <vector>
using namespace std;
#define PI ((double)3.14159265358979) 
#define ALPHA ((double)0.7) 

// Halton sequence with reverse permutation
int primes[61] = {
	2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
	83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
	191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283
};
inline int rev(const int i, const int p) {
	if (i == 0) return i; else return p - i;
}
double hal(const int b, int j) {
	const int p = primes[b];
	double h = 0.0, f = 1.0 / (double)p, fct = f;
	while (j > 0) {
		h += rev(j % p, p) * fct; j /= p; fct *= f;
	}
	return h;
}

struct Vec {
	double x, y, z; // vector: position, also color (r,g,b)
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
	inline Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	inline Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	inline Vec operator+(double b) const { return Vec(x + b, y + b, z + b); }
	inline Vec operator-(double b) const { return Vec(x - b, y - b, z - b); }
	inline Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
	inline double operator[](int axis){ return (&x)[axis]; }
	inline Vec mul(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z); }
	inline Vec norm() { return (*this) * (1.0 / sqrt(x*x + y*y + z*z)); }
	inline double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }
	Vec operator%(Vec&b) { return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x); }
};

#define MAX(x, y) ((x > y) ? x : y)

struct AABB {
	Vec min, max; // axis aligned bounding box
	inline void fit(const Vec &p)
	{
		if (p.x<min.x)min.x = p.x; // min
		if (p.y<min.y)min.y = p.y; // min
		if (p.z<min.z)min.z = p.z; // min
		max.x = MAX(p.x, max.x);
		max.y = MAX(p.y, max.y);
		max.z = MAX(p.z, max.z);
	}
	inline void reset() {
		min = Vec(1e20, 1e20, 1e20);
		max = Vec(-1e20, -1e20, -1e20);
	}
};

struct HPoint {
	Vec f, pos, nrm, flux;
	double r2;
	unsigned int n; // n = N / ALPHA in the paper
	int pix;
	bool valid;
	HPoint(){
		valid = false;
	}
};

struct KDTreeNode{
	KDTreeNode* left;
	KDTreeNode* right;
	bool isLeaf;
	std::vector<HPoint*> vps;
	AABB bbox;

	KDTreeNode(){
		left = right = NULL;
		isLeaf = false;
	}
};

unsigned int pixel_index;
KDTreeNode* root = NULL;
vector<HPoint> vpoints;
KDTreeNode* split(vector<HPoint>& vps, AABB& bbox);

void buildKdTree(){
	AABB bbox;
	bbox.reset();
	//calc bbox of visible points 
	for (int i = 0; i < vpoints.size(); ++i){
		HPoint vp = vpoints[i];
		if (!vp.valid)
			continue;
		bbox.fit(vp.pos);
	}

	root = split(vpoints, bbox);
	vpoints.clear();
}

KDTreeNode* split(vector<HPoint>& vps, AABB& bbox) {
	if (vps.size() <= 8){
		KDTreeNode* node = new KDTreeNode();
		node->isLeaf = true;
		node->bbox = bbox;
		node->vps.reserve(vps.size());
		for (int i = 0; i < vps.size(); ++i){
			HPoint* vp = new HPoint(vps[i]);
			node->vps.push_back(vp);
		}

		return node;
	}

	auto getMaxExtent = [](AABB& bbox)->int{
		Vec diag = bbox.max - bbox.min;
		if (diag.x > diag.y && diag.x > diag.z)
			return 0;
		else if (diag.y > diag.z)
			return 1;
		else
			return 2;
	};
	int axis = getMaxExtent(bbox);
	Vec center = (bbox.max + bbox.min)*0.5f;
	float sp = center[axis];
	AABB box_left, box_right;
	box_left.reset();
	box_right.reset();
	vector<HPoint> left, right;
	for (int i = 0; i < vps.size(); ++i){
		HPoint vp = vps[i];
		if (!vp.valid)
			continue;
		if (vp.pos[axis] < sp){
			box_left.fit(vp.pos);
			left.push_back(vp);
		}
		else{
			box_right.fit(vp.pos);
			right.push_back(vp);
		}
	}

	KDTreeNode* inner = new KDTreeNode();
	inner->bbox = bbox;
	inner->left = split(left, box_left);
	inner->right = split(right, box_right);
	return inner;
}

void query(KDTreeNode* root, Vec& pos, vector<HPoint*>& ret){
	float dx, dy, dz;
	AABB bbox = root->bbox;
	if (pos.x <= bbox.max.x && pos.x >= bbox.min.x)
		dx = 0.f;
	else
		dx = fabs(pos.x - bbox.min.x)>fabs(pos.x - bbox.max.x) ? fabs(pos.x - bbox.max.x) : fabs(pos.x - bbox.min.x);

	if (pos.y <= bbox.max.y && pos.y >= bbox.min.y)
		dy = 0.f;
	else
		dy = fabs(pos.y - bbox.min.y)> fabs(pos.y - bbox.max.y) ? fabs(pos.y - bbox.max.y) : fabs(pos.y - bbox.min.y);

	if (pos.z <= bbox.max.z && pos.z >= bbox.min.z)
		dz = 0.f;
	else
		dz = fabs(pos.z - bbox.min.z)> fabs(pos.z - bbox.max.z) ? fabs(pos.z - bbox.max.z) : fabs(pos.z - bbox.min.z);

	if (dx*dx + dy*dy + dz*dz > 0.16)
		return;

	if (root->isLeaf){
		for (int i = 0; i < root->vps.size(); ++i){
			HPoint* vp = root->vps[i];
			Vec dis = vp->pos - pos;
			if (vp->r2 > dis.dot(dis))
				ret.push_back(vp);
		}
	}
	else{
		if (root->left){
			query(root->left, pos, ret);
		}
		if (root->right){
			query(root->right, pos, ret);
		}
	}
}

void query(KDTreeNode* root, vector<HPoint*>& ret){
	if (root->isLeaf){
		for (int i = 0; i < root->vps.size(); ++i){
			HPoint* vp = root->vps[i];
			ret.push_back(vp);
		}
	}
	else{
		if (root->left){
			query(root->left, ret);
		}
		if (root->right){
			query(root->right, ret);
		}
	}
}

struct Ray { Vec o, d; Ray(){}; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
struct Sphere {
	double rad; Vec p, c; Refl_t refl;
	Sphere(double r_, Vec p_, Vec c_, Refl_t re_) : rad(r_), p(p_), c(c_), refl(re_){}
	inline double intersect(const Ray &r) const {
		// ray-sphere intersection returns distance
		Vec op = p - r.o;
		double t, b = op.dot(r.d), det = b*b - op.dot(op) + rad*rad;
		if (det < 0) {
			return 1e20;
		}
		else {
			det = sqrt(det);
		}
		return (t = b - det) > 1e-4 ? t : ((t = b + det)>1e-4 ? t : 1e20);
	}
};

Sphere sph[] = { // Scene: radius, position, color, material
	Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(.75, .25, .25), DIFF),//Left
	Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(.25, .25, .75), DIFF),//Right
	Sphere(1e5, Vec(50, 40.8, 1e5), Vec(.75, .75, .75), DIFF),//Back
	Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), DIFF),//Front
	Sphere(1e5, Vec(50, 1e5, 81.6), Vec(.75, .75, .75), DIFF),//Bottomm
	Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(.75, .75, .75), DIFF),//Top
	Sphere(16.5, Vec(27, 16.5, 47), Vec(1, 1, 1)*.999, SPEC),//Mirror
	Sphere(16.5, Vec(73, 16.5, 88), Vec(1, 1, 1)*.999, REFR),//Glass
	Sphere(8.5, Vec(50, 8.5, 60), Vec(1, 1, 1)*.999, DIFF) };//Middle

// tone mapping and gamma correction
int toInt(double x){
	return int(pow(1 - exp(-x), 1 / 2.2) * 255 + .5);
}

// find the closet interection
inline bool intersect(const Ray &r, double &t, int &id){
	int n = sizeof(sph) / sizeof(Sphere);
	double d, inf = 1e20; t = inf;
	for (int i = 0; i<n; i++){
		d = sph[i].intersect(r);
		if (d<t){
			t = d;
			id = i;
		}
	}
	return t<inf;
}

// generate a photon ray from the point light source with QMC
void genp(Ray* pr, Vec* f, int i) {
	*f = Vec(2500, 2500, 2500)*(PI*4.0); // flux
	double p = 2.*PI*hal(0, i), t = 2.*acos(sqrt(1. - hal(1, i)));
	double st = sin(t);
	pr->d = Vec(cos(p)*st, cos(t), sin(p)*st);
	pr->o = Vec(50, 60, 85);
}

void trace(const Ray &r, int dpt, bool m, const Vec &fl, const Vec &adj, int i)
{
	double t;
	int id;

	dpt++;
	if (!intersect(r, t, id) || (dpt >= 20))return;

	int d3 = dpt * 3;
	const Sphere &obj = sph[id];
	Vec x = r.o + r.d*t, n = (x - obj.p).norm(), f = obj.c;
	Vec nl = n.dot(r.d)<0 ? n : n*-1;
	double p = f.x>f.y&&f.x>f.z ? f.x : f.y>f.z ? f.y : f.z;

	if (obj.refl == DIFF) {
		// Lambertian

		// use QMC to sample the next direction
		double r1 = 2.*PI*hal(d3 - 1, i), r2 = hal(d3 + 0, i);
		double r2s = sqrt(r2);
		Vec w = nl, u = ((fabs(w.x)>.1 ? Vec(0, 1) : Vec(1)) % w).norm();
		Vec v = w%u, d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).norm();

		if (m) {
			// eye ray
			// store the measurment point
			HPoint hp;
			hp.f = f.mul(adj);
			hp.pos = x;
			hp.nrm = n;
			hp.flux = Vec();
			hp.r2 = 0.16f;
			hp.n = 0;
			hp.pix = pixel_index;
			hp.valid = true;
			vpoints.push_back(hp);
			return;
		}
		else
		{
			vector<HPoint*> ret;
			query(root, x, ret);
			for (int i = 0; i < ret.size(); ++i){
				HPoint* hp = ret[i];
				if ((hp->nrm.dot(n) > 1e-3)){
					double g = (hp->n + ALPHA) / (hp->n + 1.0);
					hp->r2 = hp->r2*g;
					hp->n++;
					Vec a = (hp->flux + hp->f.mul(fl)*(1. / PI))*g;
					hp->flux = (hp->flux + hp->f.mul(fl)*(1. / PI))*g;
				}
			}

			if (hal(d3 + 1, i)<p) trace(Ray(x, d), dpt, m, f.mul(fl)*(1. / p), adj, i);
		}

	}
	else if (obj.refl == SPEC) {
		// mirror
		trace(Ray(x, r.d - n*2.0*n.dot(r.d)), dpt, m, f.mul(fl), f.mul(adj), i);

	}
	else {
		// glass
		Ray lr(x, r.d - n*2.0*n.dot(r.d));
		bool into = (n.dot(nl)>0.0);
		double nc = 1.0, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;

		// total internal reflection
		if ((cos2t = 1 - nnt*nnt*(1 - ddn*ddn))<0) return trace(lr, dpt, m, fl, adj, i);

		Vec td = (r.d*nnt - n*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
		double a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : td.dot(n));
		double Re = R0 + (1 - R0)*c*c*c*c*c, P = Re; Ray rr(x, td); Vec fa = f.mul(adj);
		if (m) {
			// eye ray (trace both rays)
			trace(lr, dpt, m, fl, fa*Re, i);
			trace(rr, dpt, m, fl, fa*(1.0 - Re), i);
		}
		else {
			// photon ray (pick one via Russian roulette)
			(hal(d3 - 1, i)<P) ? trace(lr, dpt, m, fl, fa, i) : trace(rr, dpt, m, fl, fa, i);
		}
	}
}

Vec *c = new Vec[512*512];
void StartTracing(){
	// samps * 1000 photon paths will be traced
	int w = 512, h = 512, samps = 100000;

	// trace eye rays and store measurement points
	Ray cam(Vec(50, 48, 295.6), Vec(0, -0.042612, -1).norm());
	Vec cx = Vec(w*.5135 / h), cy = (cx%cam.d).norm()*.5135, vw, *temp = new Vec[w*h];
	for (int y = 0; y < h; y++){
		fprintf(stderr, "\rHitPointPass %5.2f%%", 100.0*y / (h - 1));
		for (int x = 0; x < w; x++) {
			pixel_index = x + y * w;
			Vec d = cx * ((x + 0.5) / w - 0.5) + cy * (-(y + 0.5) / h + 0.5) + cam.d;
			trace(Ray(cam.o + d * 140, d.norm()), 0, true, Vec(), Vec(1, 1, 1), 0);
		}
	}
	fprintf(stderr, "\n");

	buildKdTree();
	// trace photon rays with multi-threading
	for (int iter = 0; iter < samps; iter++) {
		printf("%d\n", iter);
		int m = 1000 * iter;
		Ray r;
		Vec f;
#pragma omp parallel for schedule(dynamic, 1)
		for (int j = 0; j < 1000; j++){
			genp(&r, &f, m + j);
			trace(r, 0, 0 > 1, f, vw, m + j);
		}

		// density estimation

		vector<HPoint*> ret;
		query(root, ret);
		for (int i = 0; i < ret.size(); ++i){
			HPoint* vp = ret[i];
			int index = vp->pix;
			Vec li = vp->flux*(1.0 / (PI*vp->r2*(iter + 1)*1000.0));
			//skip if color is not a number
			if (isnan(li.x) || isnan(li.y) || isnan(li.z))
				li = Vec(0.f);

			temp[index] = temp[index] + li;
		}

		for (int i = 0; i < 512 * 512; ++i){
			c[i] = temp[i];
			temp[i] = Vec();
		}
	}
}

float* image = new float[512 * 512 * 3];
static void display(void){
	for (int y = 0; y < 512; ++y){
		for (int x = 0; x < 512; ++x){
			int idx = (511 - y) * 512 + x;
			int idx1 = y * 512 + x;
			int r = toInt(c[idx].x);
			int g = toInt(c[idx].y);
			int b = toInt(c[idx].z);
			image[3 * idx1] = r / 255.0;
			image[3 * idx1 + 1] = g / 255.0;
			image[3 * idx1 + 2] = b / 255.0;
		}
	}
	glDrawPixels(512, 512, GL_RGB, GL_FLOAT, image);

	glutSwapBuffers();
	glutPostRedisplay();
}

int main(int argc, char *argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(512, 512);
	glutCreateWindow("Progressive Photon Mapper");
	glewInit();

	glViewport(0, 0, 512, 512);

	glutDisplayFunc(display);

	thread* th = new thread(StartTracing);
	th->detach();
	glutMainLoop();
}