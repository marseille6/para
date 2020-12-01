#include<stdio.h>
#include<math.h>
struct ball{
    double px,py;
    double vx,vy;
    double ax,ay;
}ball;
__global__ void K_main(ball *balls_in, ball *balls_out, int N, int cycle_times, int time_step){
    double r = 0.01; // r = 1 cm
    double GM = 6.67E-7;// G = 6.67 * 10 ^ -11   M = 10000 kg
    int tid = threadIdx.x;
    double delta_t = 1 / time_step;  // delta_t = 1 / time_step
    int size = int(sqrt(N));
    maxX = (size - 1) * r; // the X length of the border
    maxY = (size - 1) * r; // the Y length of the border
    __shared__ ball sh_balls[N];
    sh_balls[tid] = balls_in[tid];
    __syncthreads();
    for(int i = 0;i < cycle_times; i ++){
        compute_force(tid,sh_balls,r);
        compute_velocities(tid,sh_balls,delta_t);
        compute_position(tid,sh_balls,maxX,maxY,delta_t);
        __syncthreads();
    }
}
__device__ void compute_force(int threadIdx,ball* balls,int r){
    balls[threadIdx].ax = 0;
    balls[threadIdx].ay = 0;
    for(int i = 0;i < 64;i ++){
        if(i != threadIdx){
            double dx = balls[i].px - balls[threadIdx].px;
            double dy = balls[i].py - balls[threadIdx].py;
            double d = dx * dx + dy * dy;
            if (d < r * r) d = r * r;
            d *= sqrt(d);
            balls[threadIdx].ax += GM * dx / d;
            balls[threadIdx].ay += GM * dy / d;
        }
    }
}

__device__ compute_velocities(int threadIdx,ball balls,double delta_t){
    balls[threadIdx].vx += balls[threadIdx].ax * delta_t;
    balls[threadIdx].vy += balls[threadIdx].ay * delta_t;
}

__device__ void compute_position(int threadIdx,ball* balls,double maxX,double maxY,double delta_t){
    double px = balls[threadIdx].px + balls[threadIdx].vx * delta_t;
    if (px > maxX) balls[threadIdx].px = maxX; // declare a maxX length
    else if(px < 0) balls[threadIdx].px = 0;
    else balls[threadIdx].px = px;
    double py = balls[threadIdx].py + balls[threadIdx].vy * delta_t;
    if (py > maxY) balls[threadIdx].py = maxX; // declare a maxY length
    else if(py < 0) balls[threadIdx].py = 0;
    else balls[threadIdx].py = py;
}
int main(){
    const int balls_Num = 64;
    const cycle_times = 1000;
    const time_step = 100;
    const int balls_Size = balls_Num * sizeof(ball);
    const double r = 0.01;
    ball h_in_balls[balls_Num];
    // init the balls
    for(int i = 0;i < 8;i ++){
        for(int j = 0; j < 8; j++){
            h_in_balls[i * 8 + j].px = i * r;
            h_in_balls[i * 8 + j].py = j * r;
            h_in_balls[i * 8 + j].vx = 0;
            h_in_balls[i * 8 + j].vy = 0;
            h_in_balls[i * 8 + j].ax = 0;
            h_in_balls[i * 8 + j].ay = 0;

        }
    }
    //
    ball h_out_balls[balls_Num];
    // declare the GPU memory pointers
    ball* d_in_balls;
    ball* d_out_balls;
    // allocate GPU memory
    cudaMalloc((void**) &d_in_balls,balls_Size);
    cudaMalloc((void**) &d_out_balls,balls_Size);
    // transfer the array to GPU
    cudaMemcpy(d_in_balls,h_in_balls,balls_Size,cudaMemcpyHostToDevice);
    // launch the kernel
    K_main<<<1,balls_Num>>>(d_in_balls,d_out_balls,balls_Num,cycle_times,time_step);
    cudaMemcpy(h_out_balls,d_out_balls,cudaMemcpyDeviceToHost);
    for(int i = 0;i < balls_Num; i ++){
        printf("%f",h_out_balls.px);
        printf("%f",h_out_balls.py);
    }
    cudaFree(d_in_balls);
    cudaMalloc(d_out_balls);
    return 0;
 }
