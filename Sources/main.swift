//
//  main.swift
//  Perfect-TensorFlow-Demo-Computer Vision
//
//  Created by Rockford Wei on 2017-06-19.
//  Copyright © 2017 PerfectlySoft. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This source file is part of the Perfect.org open source project
//
// Copyright (c) 2017 - 2018 PerfectlySoft Inc. and the Perfect project authors
// Licensed under Apache License v2.0
//
// See http://perfect.org/licensing.html for license information
//
//===----------------------------------------------------------------------===//
//

import Foundation
import PerfectLib
import PerfectTensorFlow
import PerfectHTTP
import PerfectHTTPServer

typealias TF = TensorFlow


class LabelImage {
  let def: TF.GraphDef

  public init(_ model:Data) throws {
    def = try TF.GraphDef(serializedData: model)
  }

  public func match(image: Data) throws -> (Int, Int) {
    let g = try TF.Graph()
    try g.import(definition: def)
    let normalized = try constructAndExecuteGraphToNormalizeImage(g, imageBytes: image)
    let possibilities = try executeInceptionGraph(g, image: normalized)
    guard let m = possibilities.max(), let i = possibilities.index(of: m) else {
      throw TF.Panic.INVALID
    }//end guard
    return (i, Int(m * 100))
  }

  private func executeInceptionGraph(_ g: TF.Graph, image: TF.Tensor) throws -> [Float] {
    let results = try g.runner().feed("input", tensor: image).fetch("output").run()
    guard results.count > 0 else { throw TF.Panic.INVALID }
    let result = results[0]
    guard result.dimensionCount == 2 else { throw TF.Panic.INVALID }
    let shape = result.dim
    guard shape[0] == 1 else { throw TF.Panic.INVALID }
    let res: [Float] = try result.asArray()
    return res
  }//end exec
  public func constructAndExecuteGraphToNormalizeImage(_ g: TF.Graph, imageBytes: Data) throws -> TF.Tensor {
    let H:Int32 = 224
    let W:Int32 = 224
    let mean:Float = 117
    let scale:Float = 1
    let input = try g.constant(name: "input2", value: imageBytes)
    let batch = try g.constant( name: "make_batch", value: Int32(0))
    let scale_v = try g.constant(name: "scale", value: scale)
    let mean_v = try g.constant(name: "mean", value: mean)
    let size = try g.constantArray(name: "size", value: [H,W])
    let jpeg = try g.decodeJpeg(content: input, channels: 3)
    let cast = try g.cast(value: jpeg, dtype: TF.DataType.dtFloat)
    let images = try g.expandDims(input: cast, dim: batch)
    let resizes = try g.resizeBilinear(images: images, size: size)
    let subbed = try g.sub(x: resizes, y: mean_v)
    let output = try g.div(x: subbed, y: scale_v)
    let s = try g.runner().fetch(TF.Operation(output)).run()
    guard s.count > 0 else { throw TF.Panic.INVALID }
    return s[0]
  }//end normalize
}

var inceptionModel: LabelImage? = nil
var tags = [String]()

func handler(data: [String:Any]) throws -> RequestHandler {
  return {
    request, response in
    let prefix = "data:image/jpeg;base64,"
    guard let input = request.postBodyString, input.hasPrefix(prefix),
    let b64 = String(input.utf8.dropFirst(prefix.utf8.count)),
    let scanner = inceptionModel
    else {
      response.setHeader(.contentType, value: "text/json")
        .appendBody(string: "{\"value\": \"Invalid Input\"}")
        .completed()
      return
    }//end
    let jpeg = Data.From(b64)
    guard let data = Data(base64Encoded: jpeg) else {
      response.setHeader(.contentType, value: "text/json")
        .appendBody(string: "{\"value\": \"Input is not a valid jpeg\"}")
        .completed()
      return
    }//end guard

    do {
      let result = try scanner.match(image: data)
      guard result.0 > -1 && result.0 < tags.count else {
        response.setHeader(.contentType, value: "text/json")
          .appendBody(string: "{\"value\": \"what is this???\"}")
          .completed()
        return
      }//end
      let tag = tags[result.0]
      let p = result.1
      response.setHeader(.contentType, value: "text/json")
        .appendBody(string: "{\"value\": \"Is it a \(tag)? (Possibility: \(p)%)\"}")
        .completed()
      print(tag, p)
    }catch {
      response.setHeader(.contentType, value: "text/json")
        .appendBody(string: "{\"value\": \"\(error)\"}")
        .completed()
      print("\(error)")
    }
  }
}

let confData = [
  "servers": [
    [
      "name":"localhost",
      "port":8080,
      "routes":[
        ["method":"post", "uri":"/recognize", "handler":handler],
        ["method":"get", "uri":"/**", "handler":PerfectHTTPServer.HTTPHandler.staticFiles,
         "documentRoot":"./webroot",
         "allowResponseFilters":true]
      ],
      "filters":[
        [
          "type":"response",
          "priority":"high",
          "name":PerfectHTTPServer.HTTPFilter.contentCompression,
          ]
      ]
    ]
  ]
]

do {
  let modelPath = "/tmp/tensorflow_inception_graph.pb"
  let tagPath = "/tmp/imagenet_comp_graph_label_strings.txt"
  let fModel = File(modelPath)
  let fTag = File(tagPath)
  if !fModel.exists || !fTag.exists {
    let modelPath = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
    print("please download model from \(modelPath)")
    exit(-1)
  }//end if
  try fModel.open(.read)
  let modelBytes = try fModel.readSomeBytes(count: fModel.size)
  guard modelBytes.count == fModel.size else {
    print("model file reading failed")
    exit(-3)
  }//end guard
  fModel.close()
  try fTag.open(.read)
  let lines = try fTag.readString()
  tags = lines.utf8.split(separator: 10).map { String(describing: $0) }
  try TF.Open()
  inceptionModel = try LabelImage( Data(bytes: modelBytes) )
  print("library ready")
  try HTTPServer.launch(configurationData: confData)
 }catch {
  print("fault: \(error)")
}
