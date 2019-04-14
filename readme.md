## Project Overview

One of the skills that machine-learning engineers can pick up which makes them more valuable to their employers is the ability to deploy front-end user interfaces for the models that they develop.

Historically, deploying front-end user interfaces for machine learning models to the web has been a very complicated process, involving considerable domain knowledge in networking, virtualization, containerization, and cloud computing infrastructure. 

Today, it is much easier to make the results of machine learning available on the internet due to the development of the [ZEIT Now](https://zeit.co/now) global serverless deployment platform. 

In this project, we'll be deploying a [Single Shot Multibox Detector](https://www.google.com/search?q=single+shot+multibox+detector&oq=Single+Shot+Multibox+Detector&aqs=chrome.0.0j69i61j0l4.182j1j7&sourceid=chrome&ie=UTF-8) image recognition machine learning model based on [Google's MobileNetV2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html) neural network to the cloud as a [serverless lambda](https://zeit.co/docs/v2/deployments/concepts/lambdas) using [ZEIT Now](https://zeit.co/now).  

## TL;DR

1. Clone this repository to your local development environment.

`$ git clone https://github.com/jdamiba/object-detection.git`
`$ cd object-detection`

2. Deploy to `now`. 

`$ now`

3. Go to your deployment URL.

## Object Classification & Machine Learning

Getting computers to be able to better recognize objects in images is one of the most popular applications of machine learning. 

Classification algorithms take an image and return a single output- the probability distribution over the classes of objects the algorithm has been trained to know about. 

In order to train a classification algorithm, you need to provide it with a dataset of images labeled by humans. Using this training dataset, the algorithm learns to recognize the objects that have been labeled in images it has never seen. 

Classification algorithms can work well for images that only have one object of interest in them, but they are not very useful  when there is more than one object of interest.

In order to get the computer to recognize distinct objects, we need to provide it with training data with ground-truth boxes drawn around the labeled objects in the image. Using this more detailed training dataset, an object detection algorithm can predict bounding boxes around objects in images it has never seen. 

![ground truth box](https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_stop_sign.jpg)

photo credit- pyimagesearch

![example ssd output](plane.png)

photo credit- zeit blog

## Region proposal vs. fixed grid of detectors

There are two approaches to generating bounding boxes around predicted objects in images using machine learning. 

Traditional algorithms like [Faster R-CNN](https://arxiv.org/abs/1506.01497) use the region proposal method while the newer Single Shot Multibox Detector algorithm uses a fixed grid of detectors. 

Region proposal object detection algorithms operate in two steps. First, the image is broken down into regions that are likely to contain an object. 

Then, bounding box detectors are used to make a prediction in each proposed region.  

Fixed-grid object detection algorithms break the image down into equally sized regions, and bounding box detectors are used to make predictions in each region. Since these algorithms don't have to first break the image down into irregular regions, they are considered to be one-shot algorithms relative to the two-step region proposal family of algorithms. 

The major advantage of using a fixed-grid instead of proposing regions is that your bounding box detectors can be much more specialized in the former case than in the latter since they only need to worry about identifying objects in their region of the grid. Since a bounding box detector in a region proposal algorithm needs to be able to identify an object no matter where it appears in the image, it needs to be much more generalized than a bounding box detector that only has to worry about a small part of the image and can specialize in recognizing particular objects or shapes. 

As a result of these architectural improvements, fixed-grid object detection algorithms can be trained more quickly and cheaply than other image recognition algorithms. This is especially true using neural networks like Google's MobileNetV2, which was designed to be performant in the context of CPU-constrained mobile computing devices.

A pre-trained version of a fixed-grid object detection algorithm developed by Liu et. al in 2016, the [Single Shot Multibox Detector](https://arxiv.org/abs/1512.02325) model, is made [avalaible as part of Google's Tensorflow machine learning library](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models).

## Code Overview

The web application which will deploy the SSD model has six important concerns.

1. Allow users to upload an image. (FE)

2. Convert that image into a tensor. (BE)

3. Fetch a pre-trained version of SSDlite_MobileNetV2. (BE)

4. Run the tensor of the user's image through SSDlite_MobileNetV2. (BE)

5. Convert the tensor output by the neural network into x and y coordinates describing bounding boxes. (BE)

6. Draw bounding boxes around objects detected in the user's image and display the result. (FE)

Concerns 1 and 6 will be handled by the application's front-end, which will be written with HTML/CSS/JS using ZEIT's Next.js framework.

Concerns 2-5 will be handled by the application's back-end API, which will be written in JS and deployed as a serverless lambda alongside the front-end.

### 1. Image Upload (FE)
```js

```

### 2. Image => Tensor

```js

```

### 3. Fetch SSDlite_MobileNetV2

In a traditional client-server architecture, we would be able to use the ubiquitous `@tensorflow/tfjs-node` npm package in order to fetch a pre-trained version of the SSDlite_MobileNetV2 object detection machine learning model.  This package provides a JavaScript implementation of the Tensorflow API that works in the Node.js runtime environment. 

The function [`.loadModel()`](https://js.tensorflow.org/api/0.8.0/#loadModel) allows you to load a JSON object which describes a machine learning model. Helpfully, the Tensorflow library includes [several pre-trained models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). All we need to do is make a call to the Google Storage API and download the model we want into our web server's local memory.

Unfortunately, one of the limitations of serverless lambdas is that they are more memory-constrained than traditional web servers. Clocking in at over 100MB, the `@tensorflow/tfjs-node` library is too big for us to upload to a serverless lambda, which usually has a 50MB max size limit. In order to work around this limitation, we will use the [`tensorflow-lambda`](https://github.com/lucleray/tensorflow-lambda) npm package developed by [Luc Leray](https://twitter.com/lucleray), which uses Google Brotli to compress `@tensorflow/tfjs-node` to a manageable size.

Using JavaScript's new `async/await` syntax, we can use the `tensorflow-lambda` library to load our pre-trained machine learning model in our severless lambda.

Caching the pre-trained model object detection model in local storage the first time the lambda is invoked comes with some small CPU overhead but provides a significant performance benefit for users who will persist their connection with the lambda.

```js

const loadTf = require('tensorflow-lambda')

let tfModelCache;

async function loadModel() {
  try {
    const tf = await loadTf()

    if (tfModelCache) {
      return tfModelCache
    }

    tfModelCache = await tf.loadGraphModel(`${https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2}/model.json`)
    
    return tfModelCache
    
  } catch (err) {
    console.log(err)
    throw BadRequestError('The model could not be loaded')
  }
}
```

### 4. Predict Bounding Boxes

```js
async function predict(tfModel, tensor) {
  const tf = await loadTf()

  const batched = await tf.tidy(() => tensor.expandDims())
  const result = await tfModel.executeAsync(batched)
  const scores = result[0].arraySync()[0]
  const boxes = result[1].dataSync()

  batched.dispose()
  tf.dispose(result)

  return { scores, boxes }
}
```

### 5. Bounding Boxes Tensor => X,Y Coordinates

```js

```

### 6. Display Prediction (FE)

```js

```

The application's back-end will serve an object detection API at the route `/predict`.


### Front-end Structure

Our front-end interface is divided into two parts- a sidebar with an input section where images can be uploaded and a display area where the results are presented to the user.

When an image is uploaded, the front-end makes an API call to the back-end. Hopefully, it recieves as a response a JavaScript object which describes where it should draw bounding boxes.

For example, this image would return the following:

![Screenshot of the app](plane.png)

```json
[
  {
    "bbox": [{ "x": 205.61, "y": 315.36 }, { "x": 302.98, "y": 345.72 }],
    "class": "airplane",
    "score": 0.736
  }
]
```

In case users don't have an image handy, we'll provide them with options pulled from a public image API in the sidebar.

Instructions for using the app will also be in the sidebar.

## Deploying Machine Learning Algorithms

### Overview of ZEIT Now

Now is a global serverless deployment platform, which means that it allows you to deploy machine learning models to the web without having to configure and manage a traditional web server. 

It can be installed as a command line utilty using `npm` or `yarn`. 

To deploy your applications, you simply run the `now` command in the terminal. Deployment settings can be managed by adding a `now.json` file in the root of your project directory. 

Get a feel for how easy it is to deploy sites to the web using Now:

1. `mkdir my_first_app && cd my_first_app`

2. `cat > index.html`

3. `<h1>hello, world!</h1>`

4. `now`
