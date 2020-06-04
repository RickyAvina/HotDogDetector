//
//  ViewController.swift
//  HotDogDetector
//
//  Created by Enrique Aviña on 6/2/20.
//  Copyright © 2020 Enrique Aviña. All rights reserved.
//

import UIKit
import CoreML
import AVKit
import Vision
import ImageIO

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    var textLayer = CATextLayer()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
       
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        guard let captureDevice = AVCaptureDevice.default(for: .video) else { return }
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        captureSession.addInput(input)
        captureSession.startRunning()
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        
        textLayer.frame =  CGRect(x: view.bounds.midX-130, y: view.bounds.maxY-50, width: 400, height: 50)
        textLayer.fontSize = 32
        textLayer.string = "my text"
        textLayer.foregroundColor = UIColor.red.cgColor
        textLayer.contentsScale = UIScreen.main.scale
        view.layer.addSublayer(textLayer)
        
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(dataOutput)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
              
        guard let model = try? VNCoreMLModel(for: nicehotdog().model) else {
            print("Model conversion failed (might not have image input")
            return
        }
        
        let request = VNCoreMLRequest(model: model) { (finishedRequest, error) in
            // check error
            guard let results = finishedRequest.results as? [VNClassificationObservation] else { return }
            guard let firstObservation = results.first else { return }
            
            DispatchQueue.global(qos: .background).async {
                CATransaction.begin()
                CATransaction.setDisableActions(true)
                self.textLayer.string = (firstObservation.identifier == "not_hot_dog" ? "Not hot dog" : "hot dog") + ": " + String(format: "%.2f", Double(firstObservation.confidence)*100) + "%"
                CATransaction.commit()
            }

        }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }
    
}

