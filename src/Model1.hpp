#ifndef Model1_hpp
#define Model1_hpp

#include "common.hpp"
#include "file.hpp"


class Model1 : public BasicModel
{
private:
	double AnsWeight[10][784];

	double activ(double x)
	{
		double t1 = exp(x), t2 = exp(-x);
		return (t1 + t2) / 2.0;
	}

	double grad(double c, double x0)
	{
		double tmp = exp(c * x0);
		double tmp2 = 1.0 / tmp;
		return (tmp - tmp2) / 2.0;
	}

public:
	Model1(string para1,string para2,int para3,string para4,string para5,int para6): BasicModel(para1,para2,para3,para4,para5,para6){
		//带参数的应该被随机初始化，避免陷入驻点
		std::mt19937 rng(1);
		std::uniform_real_distribution<double> dist(-0.01, 0.01);
		for (int i = 0; i < 10; ++i)
			for (int j = 0; j < 784; ++j)
				AnsWeight[i][j] = dist(rng);
	}

	Model1(){
		
	}

	bool input()
	{
		ifstream in = open_read(HDpath);
		if (!in.is_open())
		{
			cerr << "input file open failed!" << endl;
			return 1;
		}
		bool BIF=BasicModel::input(in); // 先调用基类的输入
		//BIF=basic input flag

		string weightline;
		getline(in,weightline);//感谢grok提醒 这一行是为了跳过.HD的weight字段

		if(BIF==1)return 1;
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 784; j++)
				in >> AnsWeight[i][j];
		}
		return 0;
	}

	bool output()
	{
		ofstream out = open_write(HDpath);
		if (!out.is_open())
		{
			cerr << "output file open failed!" << endl;
			return 1;
		}
		// 先调用基类的输出
		BasicModel::output(out);
		out << setprecision(numeric_limits<double>::max_digits10);
		out << "weight" << endl;
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 784; j++)
				out << AnsWeight[i][j] << " ";
		}
		return 0;
	}



	int train(int batch)
	{
	//返回值由void改成int 返回0代表正常 1代表训练集不足，完成训练 2代表异常
	//3代表已经完成训练,train()指令不生效

		//首先校验剩余训练集数量是否合格，以及是否完成
		//这部分应该是通用的吧，应该整合到BaiscModel的train()里面
		if(train_finished)return 3;
		int LeftNums=max_train_num-train_img_num;
		int size=0;
		if(LeftNums>=batch)
		{
			size=batch;
		}
		else{
			size=LeftNums;
		}

		//引用file.hpp和FilePath.hpp获取image和label
		//当然，这里FilePath.hpp的调用是间接的
		
		image *front=nullptr;
		inputImg(train_img_path,size,train_img_num,front);

		int *label=nullptr;
		inputLabel(train_label_path,size,train_img_num,label);


		double delta[10][784] = {0.0};
		const double alpha = 0.01; // 学习率

		
		for(int i=0;i<size;i++)
		{
			//正向传播
			possible basic1;
			for (int j = 0; j < 10; j++)
				basic1[j] = 0;

			for (int j = 0; j < 10; j++)
			{
				double &solvep = basic1[j];
				for (int k = 0; k < 784; k++)
				{
					double tmp = AnsWeight[j][k] * front[i][k];
					solvep += activ(tmp);
				}
			}

			possible basic2 = Normal(basic1);

			//反向传播
			for (int j = 0; j < 10; j++)
			{
				double loss = basic2[j] - (j == label[i] ? 1.0 : 0.0);
				for (int k = 0; k < 784; k++)
				{
					double this_color = front[i][k];
					double wgrad = grad(this_color, AnsWeight[j][k]);
					delta[j][k] += loss * wgrad * this_color;
				}
			}
		}

		for (int j = 0; j < 10; j++)
		{
			for (int k = 0; k < 784; k++)
			{
				AnsWeight[j][k] -= delta[j][k] * alpha / size;
			}
		}


		train_img_num+=size;
		if(train_img_num==max_train_num)
			train_finished=1;

		delete [] front;
		delete [] label;

		
		if(size==batch){
			return 0;
		}
		else{
			return 1;
		}
	}

	

	possible solve(const image &input)
	{//用来测试单张图片
		possible basic1;
		for (int i = 0; i < 10; i++)
			basic1[i] = 0;

		for (int i = 0; i < 10; i++)
		{
			double &solvep = basic1[i];
			for (int j = 0; j < 784; j++)
			{
				double tmp = AnsWeight[i][j] * input[j];
				solvep += activ(tmp);
			}
		}

		possible basic2 = Normal(basic1);
		return basic2;
	}

	double GetTest(){//返回正确率
	//对当前测试集，实际上是验证集进行测试
		int cnt_right=0;	

		int now_test_num=0;
		while(now_test_num<max_train_num){
			int LeftNums=max_train_num-now_test_num;
			int size=0;
			if(LeftNums>=100)
			{
				size=100;
			}
			else{
				size=LeftNums;
			}

			image *front=nullptr;
			inputImg(train_img_path,size,now_test_num,front);

			int *label=nullptr;
			inputLabel(train_label_path,size,now_test_num,label);

			for(int i=0;i<size;i++){
				int ans=*(label+i);
				possible work=solve(*(front+i));
				int rk=0;double maxwork=work[0];
				for(int i=1;i<10;i++){
					if(work[i]>maxwork)
					{
						maxwork=work[i];
						rk=i;
					}
				}
				if(rk==ans){
					cnt_right++;
				}
			}

			delete [] front;
			delete [] label;

			now_test_num+=size;
		}
		
		return double(cnt_right)/double(max_train_num);
	}

};

#endif
