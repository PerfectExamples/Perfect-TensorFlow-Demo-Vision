// swift-tools-version:4.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PerfectTensorFlowDemo",
    dependencies: [
        .package(url: "https://github.com/PerfectlySoft/Perfect-HTTPServer.git", from: "3.0.0"),
        .package(url: "https://github.com/PerfectlySoft/Perfect-TensorFlow.git", from: "1.4.0"),
    ],
    targets: [
        .target(
            name: "PerfectTensorFlowDemo",
            dependencies: ["PerfectHTTPServer", "PerfectTensorFlow"]),
    ]
)
