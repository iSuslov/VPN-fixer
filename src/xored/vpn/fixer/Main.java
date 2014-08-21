/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package xored.vpn.fixer;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import static xored.vpn.fixer.Main.logError;
import org.apache.commons.io.FileUtils;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.imgrec.ImageRecognitionPlugin;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import static org.opencv.highgui.Highgui.*;
import org.opencv.imgproc.Imgproc;

class TokenDetector {

	
	
	//String workspace = "C:/token_recognition/";
	String workspace = "C:/Users/User/Desktop/Token Detector/";
	String temp = workspace + "Temp/";

	public boolean run() throws IOException, InterruptedException {
		saveImage();
		Mat source = Highgui.imread(workspace + "original.jpg", Highgui.CV_LOAD_IMAGE_COLOR);

		Mat destination = new Mat(source.rows(), source.cols(), source.type());

		List<Mat> mats = new ArrayList<>();
		mats.add(new Mat());
		mats.add(new Mat());
		mats.add(new Mat());
		Core.split(source, mats);

		
		mats.get(1).convertTo(destination, -1, 2);
		imwrite(workspace + "green-ajusted.jpg", destination);

		Mat image = Highgui.imread(workspace + "green-ajusted.jpg", Imgproc.COLOR_RGB2GRAY);

		Rect rect = new Rect(176, 265, 178, 52);
		Mat imageB = threshold(image.submat(rect), 15, 2);
		

		int iWidth = 28;
		int iHeight = 45;
		List<Mat> matList = Arrays.asList(imageB,
				imageB.submat(new Rect(0, 0, iWidth, iHeight)),
				imageB.submat(new Rect(28, 2, iWidth, iHeight)),
				imageB.submat(new Rect(54, 2, iWidth, iHeight)),
				imageB.submat(new Rect(95, 5, iWidth, iHeight)),
				imageB.submat(new Rect(122, 6, iWidth, iHeight)),
				imageB.submat(new Rect(150, 7, iWidth, iHeight)));
		imwrite(temp + "1.jpg", matList.get(1));
		imwrite(temp + "2.jpg", matList.get(2));
		imwrite(temp + "3.jpg", matList.get(3));
		imwrite(temp + "4.jpg", matList.get(4));
		imwrite(temp + "5.jpg", matList.get(5));
		imwrite(temp + "6.jpg", matList.get(6));

		Mat m = addTo(source, matList);
		Core.rectangle(m, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255));

		String code = recognize();
		imwrite(workspace + "ResultsDebug/" + code + ".jpg", m);

		Main.log("Using code: " + code);
		Main.log("Using username: " + Main.username);
		Main.log("Using pin: " + Main.pin);

		String command = Main.nclauncher + " -url https://vpn.spirent.com/xored -r \"Contractor - Xored\" -u " + Main.username + " -p " + Main.pin + code;
		String response = Main.execCmd(command);
		Main.log(response);
		return response.indexOf("is already running") >= 0;
	}



	private Mat addTo(Mat matA, List<Mat> mats) {
		Mat m = new Mat(matA.rows(), matA.cols(), matA.type());
		for (int i = 0; i < matA.rows(); i++) {
			for (int j = 0; j < matA.cols(); j++) {
				m.put(i, j, matA.get(i, j));
			}
		}
		int xOffset = 0;
		for (Mat mat : mats) {
			for (int i = 0; i < mat.rows(); i++) {
				for (int j = 0; j < mat.cols(); j++) {
					double[] v = mat.get(i, j);
					if (v.length == 1) {
						double[] v1 = {v[0], v[0], v[0]};
						m.put(i, j + xOffset, v1);
					} else {
						m.put(i, j + xOffset, v);
					}
				}
			}
			xOffset += mat.cols() + 10;
		}

		return m;
	}

	private Mat threshold(Mat image, int blockSize, double c) {
		Mat imageHSV = new Mat(image.size(), Core.DEPTH_MASK_8U);
		Mat imageBlurr = new Mat(image.size(), Core.DEPTH_MASK_8U);
		Mat imageA = new Mat(image.size(), Core.DEPTH_MASK_ALL);
		Imgproc.cvtColor(image, imageHSV, Imgproc.COLOR_BGR2GRAY);
		Imgproc.GaussianBlur(imageHSV, imageBlurr, new Size(11, 11), 0);
		Imgproc.adaptiveThreshold(imageBlurr, imageA, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, blockSize, c);
		return imageA;
	}

	private void saveImage() throws IOException {
		FileUtils.copyURLToFile(new URL(Main.tokenUrl), new File(
				workspace + "original.jpg"));
	}

	private String recognize() {
		NeuralNetwork nnet = NeuralNetwork.load(this.getClass().getResourceAsStream("50.nnet")); 
		
		ImageRecognitionPlugin imageRecognition = (ImageRecognitionPlugin) nnet.getPlugin(ImageRecognitionPlugin.class); 
		String result = "";
		try {
			
			for (int i = 1; i <= 6; i++) {
				Map<String, Double> output = imageRecognition.recognizeImage(new File(temp + i + ".jpg"));
				result += getMostProbable(output);
			}
		} catch (IOException ioe) {
			//да и хуй с тобой
		}
		return result;
	}

	private String getMostProbable(Map<String, Double> output) {
		Map<String, Double> probabilities = new HashMap<>();
		for (String k : output.keySet()) {
			String key = key(k);
			if (!probabilities.containsKey(key)) {
				probabilities.put(key, 0.0);
			}
			probabilities.put(key, probabilities.get(key) + output.get(k));
		}
		String bestV = "";
		Double bestP = 0.0;
		for (String k : probabilities.keySet()) {
			if (probabilities.get(k) >= bestP) {
				bestV = k;
				bestP = probabilities.get(k);
			}
		}
		return bestV;
	}

	private String key(String key) {
		return key.replaceAll(" .*", "");
	}
}

public class Main {
	private static final String USERNAME_KEY = "vpn.token.username";
	private static final String PIN_KEY = "vpn.token.pin";
	private static final String NCLAUNCHER_KEY = "vpn.token.nclauncher";
	private static final String TOKEN_URL_KEY = "vpn.token.url";
	public static String username;
	public static String pin;
	public static String nclauncher;
	public static String tokenUrl;

	public static void main(String[] args) throws InterruptedException, IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		username = System.getProperty(USERNAME_KEY);
		pin = System.getProperty(PIN_KEY);
		nclauncher = System.getProperty(NCLAUNCHER_KEY);
		tokenUrl = System.getProperty(TOKEN_URL_KEY);
		
		if (username == null || username.trim().isEmpty() || pin == null || pin.trim().length() != 4) {
			logError("Username or pin is not set, exiting...");
			return;
		}
		if (nclauncher == null) {
			logError("Path to nclauncher.exe is not set, exiting...");
			return;
		}
		if (tokenUrl == null) {
			logError("Url to token is not set, exiting...");
			return;
		}
		int tryNumber = 0;
		TokenDetector td = new TokenDetector();
		while (true) {
			if (isVpnDown()) {
				if (tryNumber == 1) {
					logError("Last try was unsuccessful. Will try one more time...");
				} else if (tryNumber == 2) {
					logError("Last two attempts to restore VPN connection were unsuccessful. Have to stop..");
					return;
				} else {
					log("VPN is down! Trying to restore...");
				}
				log("Killing Network Connect ...");
				log(execCmd("taskkill /F /IM dsNetworkConnect.exe"));
				
				try {
					if (td.run()) {
						Thread.sleep(60000);
					}
				} catch (Throwable e) {
					// ignore
				}
				tryNumber++;
				Thread.sleep(10000);
			} else {
				tryNumber = 0;
			}
		}

	}

	public static void log(String msg) {
		System.out.println(new Date().toString() + " -[INFO]- " + msg);
	}

	public static void logError(String msg) {
		System.out.println(new Date().toString() + " -[ERROR]- " + msg);
	}

	public static String execCmd(String cmd) throws java.io.IOException, InterruptedException {
		Process p = Runtime.getRuntime().exec(cmd);
		p.waitFor();
		InputStream io;
		if (p.exitValue() == 0) {
			io = p.getInputStream();
		} else {
			io = p.getErrorStream();
		}
		Scanner s = new Scanner(io).useDelimiter("\\A");
		return s.hasNext() ? s.next() : "";

	}

	private static boolean isVpnDown() throws java.io.IOException, InterruptedException {
		boolean isDown = execCmd("ping jenkins-ito.spirenteng.com").indexOf("TTL=") < 0
				&& execCmd("ping origin.spirenteng.com").indexOf("TTL=") < 0;
		return isDown;
	}

}
