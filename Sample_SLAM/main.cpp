#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <io.h>
#include<opencv2/core/core.hpp>
#include<System.h>
#include <omp.h>


using namespace std;

template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

void LoadImages(const string& strImagePath, const string& strPathTimes, vector<string>& vstrImages, vector<double>& vTimeStamps);

void LoadIMU(const string& strImuPath, vector<double>& vTimeStamps, vector<cv::Point3f>& vAcc, vector<cv::Point3f>& vGyro);

int main(int argc, char** argv)
{

	// Retrieve paths to images
	vector<string> vstrImageFilenames;
	vector<double> vTimestamps;
	vector<double> vTimestampsImu;
	vector< cv::Point3f > vAcc, vGyro;

	int first_imu = 0;

	//bool bReuseMap = false;
	//string mapFile = "./map/Test_ORB_SLAM_3.bin";
	string path_to_vocabulary = "./Dataset/orbvoc_9_5.bin";
	//string path_to_settings = "./Dataset/CameraSettings_SM-A9200.yaml";
	string path_to_settings = "./Dataset/CameraSettings_EuRoc.yaml";
	string path_to_imu_config = "./Dataset/IMU_Config.yaml";

	//string path_to_sequence = "E:/data/SLAM/TestIMU/sequence_300hz/out/cam0/data";
	//string path_to_timestamps = "E:/data/SLAM/TestIMU/sequence_300hz/out/cam0/data.csv";
	//string path_to_IMU = "E:/data/SLAM/TestIMU/sequence_300hz/out/imu0/data.csv";

	string path_to_sequence = "E:/data/SLAM/EuRoC/MH_02_easy/mav0/cam0/data";
	string path_to_timestamps = "E:/data/SLAM/EuRoC/MH_02_easy/mav0/cam0/data.csv";
	string path_to_IMU = "E:/data/SLAM/EuRoC/MH_02_easy/mav0/imu0/data.csv";

	//string path_to_sequence = "E:/data/SLAM/EuRoC/MH_03_medium/mav0/cam0/data";
	//string path_to_timestamps = "E:/data/SLAM/EuRoC/MH_03_medium/mav0/cam0/data.csv";
	//string path_to_IMU = "E:/data/SLAM/EuRoC/MH_03_medium/mav0/imu0/data.csv";


	if (argc >= 2)
		path_to_vocabulary = argv[1];
	if (argc >= 3)
		path_to_settings = argv[2];
	if (argc >= 4)
		path_to_sequence = argv[3];


	LoadImages(path_to_sequence, path_to_timestamps, vstrImageFilenames, vTimestamps);
	LoadIMU(path_to_IMU, vTimestampsImu, vAcc, vGyro);

	// Find first imu to be considered, supposing imu measurements start first
	while (vTimestampsImu[first_imu] <= vTimestamps[0])
		first_imu++;
	first_imu--;

	int nImages = vstrImageFilenames.size();

	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	
	ORB_SLAM2::System SLAM(path_to_vocabulary, path_to_settings, ORB_SLAM2::System::MONOCULAR, true);

	ORB_SLAM2::ConfigParam Param(path_to_imu_config);
	SLAM.SetConfigParam(&Param);
	SLAM.SetMonoVIEnable(true);

	// Vector for tracking time statistics
	vector<float> vTimesTrack;
	vTimesTrack.resize(nImages);

	cout << endl << "-------" << endl;
	cout << "Start processing sequence ..." << endl;
	cout << "Images in the sequence: " << nImages << endl << endl;

	// Main loop
	cv::Mat im;
	double tframe = 0.0;
	double start = omp_get_wtime();

	std::vector<ORB_SLAM2::IMUData> imu_data;

	for (int ni = 0; ni < nImages; ni++)
	{
		// Read image from file
		im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_COLOR);
		printf("%d/%d:%s\n", ni, nImages, vstrImageFilenames[ni].c_str());
		//double tframe = vTimestamps[ni];

		if (im.empty())
		{
			cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
			return 1;
		}

		//cv::resize(im, im, cv::Size(480, 640));
		cv::resize(im, im, cv::Size(752, 480));

		//if (ni > 1000 && ni < 1100)
		//	cv::resize(im, im, cv::Size(480 / 2, 640 / 2));

		imu_data.clear();

		if (ni > 0)
		{
			while (vTimestampsImu[first_imu] <= vTimestamps[ni])
			{
				imu_data.push_back(ORB_SLAM2::IMUData(vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z, 
													  vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
													  vTimestampsImu[first_imu]));
				first_imu++;
			}
		}

		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

		cv::Mat pose = SLAM.TrackMonoVI(im, imu_data, vTimestamps[ni]);

		if (!pose.empty())
		{
			printf("i = %d\n", ni);
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					printf("%12.5f", pose.at<float>(i, j));
				}
				printf("\n");
			}
		}

		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
		double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

		vTimesTrack[ni] = ttrack;

	}
	printf("all images have been processed\n");
	// Stop all threads
	system("pause");
	SLAM.Shutdown();


	// Tracking time statistics
	sort(vTimesTrack.begin(), vTimesTrack.end());
	float totaltime = 0;
	for (int ni = 0; ni < nImages; ni++)
	{
		totaltime += vTimesTrack[ni];
	}
	cout << "-------" << endl << endl;
	cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
	cout << "mean tracking time: " << totaltime / nImages << endl;

	//SLAM.Clear();

	return 0;
}


void LoadImages(const string& path, vector<string>& vstrImageFilenames)
{
	//文件句柄  
	long long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))  //如果是目录,迭代之
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					LoadImages(p.assign(path).append("/").append(fileinfo.name), vstrImageFilenames);
			}
			else  //如果不是,加入列表  
			{
				std::string fname(fileinfo.name);
				std::string imgType = fname.substr(fname.rfind("."), fname.length());
				if (imgType == ".jpg" || imgType == ".jpeg" || imgType == ".JPG" || imgType == ".JPEG" || imgType == ".bmp" || imgType == ".BMP" || imgType == ".png" || imgType == ".PNG" || imgType == ".avi" || imgType == ".AVI" || imgType == ".ppm" || imgType == ".PPM")
				{
					vstrImageFilenames.push_back(p.assign(path).append("/").append(fileinfo.name));
					//std::cout << "find pic :" << vstrImageFilenames.size() << std::endl;
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	std::cout << "find pic :" << vstrImageFilenames.size() << std::endl;

}

void LoadImages(const string& strImagePath, const string& strPathTimes, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
	//Load function for EuRoc data
	ifstream fTimes;
	fTimes.open(strPathTimes.c_str());
	vTimeStamps.reserve(8000);
	vstrImages.reserve(8000);

	while (!fTimes.eof())
	{
		string s;
		getline(fTimes, s);
		if (s[0] == '#')
			continue;

		if (!s.empty())
		{
			int pos = s.find(',');
			string item;
			double time;
			item = s.substr(0, pos);
			time = stod(item);
			string name = "/" + s.substr(pos + 1, pos * 2 + 4);

			vstrImages.push_back(strImagePath + name);
			vTimeStamps.push_back(time / 1e9);
		}
	}
}

void LoadIMU(const string& strImuPath, vector<double>& vTimeStamps, vector<cv::Point3f>& vAcc, vector<cv::Point3f>& vGyro)
{
	ifstream fImu;
	fImu.open(strImuPath.c_str());
	vTimeStamps.reserve(50000);
	vAcc.reserve(50000);
	vGyro.reserve(50000);

	while (!fImu.eof())
	{
		string s;
		getline(fImu, s);
		if (s[0] == '#')
			continue;

		if (!s.empty())
		{
			string item;
			size_t pos = 0;
			double data[7];
			int count = 0;
			while ((pos = s.find(',')) != string::npos) {
				item = s.substr(0, pos);
				data[count++] = stod(item);
				s.erase(0, pos + 1);
			}
			item = s.substr(0, pos);
			data[6] = stod(item);

			vTimeStamps.push_back(data[0] / 1e9);
			vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
			vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
		}
	}
}
