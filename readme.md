## Deployment Instructions

1. Clone this repository.

`$ git clone https://github.com/jdamiba/object-detection.git`
`$ cd object-detection`

2. Deploy to now. 

`$ now`

3. Go to your deployment URL.

## Documentation

### Overview

Image recognition is a key application of machine learning because getting a computer to be able to independently make decisions based on information gathered from still images and/or video is of enormous econocmic and social value.

Of course, photo-sensitive sensors have been in industrial use for decades. For example, this tomato sorting machine uses optical sensors to reject unripe fruit. 

[![tomato sorter](https://img.youtube.com/vi/j4RWJTs0QCk/0.jpg)](https://www.youtube.com/watch?v=j4RWJTs0QCk)

Modern image recognition systems are capable of much more complex analysis, and are networked with other computer systems. 

[![facial recognition](https://img.youtube.com/vi/Fq1SEqNT-7c/0.jpg)](https://www.youtube.com/watch?v=Fq1SEqNT-7c)

Implementing image recognition algorithms and deploying them to the Internet used to be a complicated process, but recently released libraries and frameworks make it much easier.

### The project

Our goal is to deploy a website where users can upload images for analysis by a pre-trained machine learning image recognition model.

A great tool for doing this is Google's Tensorflow library, which offers a variety of pre-trained models and an easily accessible JavaScript API. 

Although power users could consume our API using command line tools like `curl`, in order for users to interact with our API easily we need to build a front-end user interface which will run in web browsers.

I like using the Next.js framework because it is built on top of React.js and also tightly integrated with the Now serverless application deployment platform.

### API Structure

Our image recognition API will have one publicly accessible HTTP endpoint: `/predict`. 

If the request URL includes a `.jpg` or `.png` image, we will load both a pre-trained image recognition algorithm and the image into the client's memory. Then we'll send a response back with bounding boxes.

Otherwise, we will reply with an error message.

```js
if (req.url === '/api/predict') {
      const tf = await loadTf()
      const tfModel = await loadModel()

      const { type: mimeType } = contentType.parse(req)

      if (CONTENT_TYPES_IMAGE.includes(mimeType)) {
        const buf = await buffer(req, { limit: '5mb' })
        const { tensor, width, height } = await imgToTensor(buf)

        const { scores, boxes } = await predict(tfModel, tensor)

        const bboxes = await tensorsToBBoxes({ scores, boxes, width, height })

        return send(res, 200, bboxes)
      }

      throw BadRequestError('Only images are supported at the moment')
    }
```

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

If you want to read more about how this was made, [read the detailed article about it on ZEIT's blog](https://zeit.co/blog/serverless-machine-learning).
