#include<time.h>
#include<stdlib.h>
int main()
{
	const int Num = 64;
	const int length = 8;
	const long divNum = 10000000000;
	srand((unsigned int)time(0));//修改种子
	for(int i = 0; i < Num; i++){
	    int x = 0;
	    int y = 0;
	    for(int j = 0; j < length; j++){
	        if(j == 0){
	            x = rand() % 8;
	            y = rand() % 8;
	        }else {
	            x = rand() % 10 + x * 10;
		        y = rand() % 10 + y * 10;
		    }
		}
		double sumX = x / divNum;
		double sumY = y / divNum;
		printf("the position of the ball[%d] is (%lf,%lf)  ",i,sumX,sumY);
	}
}
