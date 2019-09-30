# RSA_face_detection
#### RSA for face detection caffe c++ version

Caffe C++ version codebase for *Recurrent Scale Approximation for Object Detection in CNN* published at **ICCV 2017**, [[arXiv\]](https://arxiv.org/abs/1707.09531). Here we just offer test code.

#### demo result
![](https://github.com/ZhouKai90/RSA_face_detection/blob/master/images/landmark/demo2.jpg)
![](https://github.com/ZhouKai90/RSA_face_detection/blob/master/images/landmark/demo1.jpg)

#### How to use

Making sure your environment can compile caffe with GPU successful.

`mkdir build`

`cd build && cmake.. && make && cd ..`

`./demo/target images`

