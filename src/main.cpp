#include <deque>
#include <iostream>
#include <mutex>
#include <memory>
#include <cmath>
#include <thread>
#include <vector>
#include <cstring>
#include <queue>

#ifdef __cplusplus
extern "C"{
#endif


typedef struct TimeStamp_ {
  uint32_t sec;
  uint32_t nsec;
}TimeStamp;

typedef struct DataHeader_ {
  TimeStamp stamp;
  uint32_t seq;
  int8_t frameId[32];
}DataHeader;


typedef struct Point3FWithCovariance_ {
  float x;
  float y;
  float z;
  float covariance[9];
}Point3FWithCovariance;


typedef struct Quaternion4FWithCovariance_ {
  float x;
  float y;
  float z;
  float w;
  float covariance[9];
}Quaternion4FWithCovariance;

typedef struct Pose_ {
  Point3FWithCovariance position;
  Quaternion4FWithCovariance orientation;
}Pose;


typedef struct Velocity_ {
  Point3FWithCovariance linear;
  Point3FWithCovariance angular;
}Velocity;

typedef struct Acceleration_ {
  Point3FWithCovariance angular;
  Point3FWithCovariance linear;
}Acceleration;

typedef struct LocationData_ {
  DataHeader header;
  Pose pose;
  Velocity velocity;
  Acceleration acceleration;
  float odometry;
  uint8_t utmLongitudeZone;
  uint8_t locationState;
  uint8_t type;
  uint8_t resetCount;
  int8_t utmLatitudeZone;
}LocationData;

#ifdef __cplusplus
}
#endif




struct LocationPerception {
    LocationData location;
    bool islocationUpdate;
};


std::mutex m_mutexLocation;

int main() {

    std::deque<LocationPerception> b;

    std::thread t([&]() {
        while (true) {
            {
                {
                    std::lock_guard<std::mutex> guard(m_mutexLocation);
                    LocationPerception location_data;
                    while (b.size() >= 100) {
                        b.pop_front();
                    }
                    b.push_back(location_data);
                }
                b.back().islocationUpdate = true;
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
    });

    std::thread t2([&]() {
        while (true) {
            {
                std::lock_guard<std::mutex> guard(m_mutexLocation);
                std::deque<LocationPerception> a;
                a.resize(12);
                static int index_1 = 0;
                ++index_1;
                if (index_1 == 50) {
                    index_1 = 0;
                    b.clear();
                }
                a.assign(b.begin(), b.end());
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    // std::thread t3([&]() {
    //     while (true) {
    //         {
    //             std::lock_guard<std::mutex> guard(m_mutexLocation);
    //             std::deque<LocationPerception> c;
    //             c.resize(100);
    //             static int index_2 = 0;
    //             ++index_2;
    //             if (index_2 == 50) {
    //                 index_2 = 0;
    //                 b.clear();
    //             }
    //             c.assign(b.begin(), b.end());
    //         }
    //         // std::this_thread::sleep_for(std::chrono::milliseconds(2));
    //     }
    // });

    t.join();
    t2.join();
    // t3.join();


    return 0;
}