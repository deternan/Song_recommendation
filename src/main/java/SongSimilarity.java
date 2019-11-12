
/*
 * word2vec at Song Recommendation (Similar)
 * 
 * version: November 11, 2019 04:40 PM
 * Last revision: November 12, 2019 12:09 PM
 * 
 * Author : Chao-Hsuan Ke
 * Institute: Delta Research Center
 * Company : Delta Electronics Inc. (Taiwan)
 * 
 */


import com.mayabot.blas.Vector;
import com.mayabot.mynlp.fasttext.FastText;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.util.ArrayList;

public class SongSimilarity
{
	FastText fastText_zh;
	
	private int wordim = 300;
	private String sourcebinPath = "D:\\data model\\";					// bin source folder
	private String sourcebinName = "wiki.zh.Chinese.bin";				// bin folder
	private String modelFolder_zh = "cc.zh.Chinese.model";
	
	// Read file
	private String folder = "D:\\Phelps\\GitHub\\Song_recommendation\\data\\";		// Text file folder
	private String file = "lyrics_word_net_mayday.dataset";
	//private String file = "single_song.txt";										// file name
	private BufferedReader bfr;	
	
	private ArrayList averageValue = new ArrayList();
	
	// Write output
	BufferedWriter writer;
	private String output_folder = folder;
	private String output_file = "lyrics_word2vec.vec";
	private String title_file = "lyrics_mayday.txt";
	private ArrayList<String> termTitle = new ArrayList<String>();
	
	
	private String inputStr = "是 你 的 形影 叫 我 逐天 作眠 夢 夢中 可愛 的 人 伊 不是 別人 我 的 每 一天 一分鐘 也 不當 輕鬆 你 是 阮愛的 人 將阮來 戲弄 九月 的 風 在 吹 那會 寒到 心肝 底 希望 變 無望 決定 我 的 一 世人 I love you 無望 你 甘是 這款 人 沒 法度 來作 陣 也 沒 法度 將我放 I love you 無望 我 就是 這款 人 我 身邊 沒半項 只有 對 你 的 想 陪伴 我 的 每 一天";
	private double[] inputValue = new double[wordim];
	private double cosineSimilarity_vale;
	//private java.util.Vector cosineSimilarity_vale_vec;
	private double[] vector_list_1;	
	private double[] vector_list_2;	
	// Similarity
	private java.util.Vector allValue = new java.util.Vector();
	
	
	// Sort
	private ArrayList<Double> termScore = new ArrayList<Double>();
	
	public SongSimilarity() throws Exception
	{
		// load bin		
			// load model (Chinese)
			fastText_zh = FastText.loadModel(sourcebinPath + modelFolder_zh, true);
		
		//writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output_folder + output_file), "utf-8"));			
		// read data & output vec
//		Read_text();			
//		writer.close();
		
		// Similarity calculation
		Str2Vec(inputStr);
		Read_VecValue();
		Read_Title();
		Comparison(inputValue, allValue);
		BubbleSort_ArrayList(termScore, termTitle);
		
		System.out.println(inputStr);
		for(int i=0; i<termTitle.size(); i++)
		{
			System.out.println(termTitle.get(i)+"	"+termScore.get(i));
		}
	}
	
	private void Read_text() throws Exception
	{
		String Line = "";
		FileReader fr = new FileReader(folder + file);
		bfr = new BufferedReader(fr);
				
		int count = 0;
		String valueStr = "";
		while((Line = bfr.readLine())!=null)
		{					
			System.out.println(count+++"	"+Line);
			averageValue.clear();
			valueStr = "";
			Text_Parsing(Line);
			for(int j=0; j<wordim; j++) {				
				if(j == wordim-1) {
					valueStr += String.valueOf(averageValue.get(j));
				}else {
					valueStr += String.valueOf(averageValue.get(j)) + ",";
				}				
			}
			writer.write(valueStr+"\n");			
		}
		fr.close();
		bfr.close();
		
	}
	
	private void Text_Parsing(String inputStr)
	{
		String temp[];
		temp = inputStr.split(" ");
		//System.out.println(temp.length);
		
		// Word2Vector
		Word2Vector(temp);
	}
	
	private void Word2Vector(String strTemp[])
	{
		double[] averageValueTmp = new double[wordim];
		for(int i=0; i<strTemp.length; i++){
			Vector vecTmpzh = fastText_zh.getWordVector(strTemp[i].toString());
			//System.out.println(strTemp[i]+"	"+vecTmpzh);
			for(int j=0; j<wordim; j++) {
				averageValueTmp[j] += vecTmpzh.get(j);
			}
		}
		
		// average		
		for(int j=0; j<wordim; j++) {		
			averageValue.add(averageValueTmp[j]/strTemp.length);						
		}		
	}

	private void Str2Vec(String inputStr)
	{
		String strTemp[];
		strTemp = inputStr.split(" ");
		
		//inputValue
		for(int i=0; i<strTemp.length; i++){
			Vector vecTmpzh = fastText_zh.getWordVector(strTemp[i].toString());
			for(int j=0; j<wordim; j++) {				
				inputValue[j] = vecTmpzh.get(j);
			}
		}
	}
	
	private void Read_VecValue() throws Exception 
	{
		String Line = "";
		FileReader fr = new FileReader(output_folder + output_file);
		bfr = new BufferedReader(fr);
											
		while((Line = bfr.readLine())!=null)
		{						
			allValue.add(Line);
		}		
		
		fr.close();
		bfr.close();
	}
	
	private void Read_Title() throws Exception
	{
		String Line = "";
		FileReader fr = new FileReader(output_folder + title_file);
		bfr = new BufferedReader(fr);
						
		String temp[];
		while((Line = bfr.readLine())!=null)
		{						
			//allValue.add(Line);
			temp = Line.split(" ");
			termTitle.add(temp[0]);
		}		
		
		fr.close();
		bfr.close();
	}
		
	private void Comparison(double[] targetVector, java.util.Vector allValue)
	{			
		ArrayList<double[]> tfidfDocsVector = new ArrayList<double[]>();
		double[] temp;
		String[] QQ;
		for(int i=0; i<allValue.size(); i++)
		{
			temp = new double[wordim];			
			QQ = allValue.get(i).toString().split(",");
			for(int j=0; j<wordim; j++) {
				temp[j] = Double.parseDouble(QQ[j]);
			}
			tfidfDocsVector.add(temp);
		}		
							
		vector_list_1 = targetVector;

		for (int j = 0; j < tfidfDocsVector.size(); j++) {
			vector_list_2 = tfidfDocsVector.get(j);

			cosineSimilarity_vale = CosineSimilarity(vector_list_1, vector_list_2);
			if (Double.isNaN(cosineSimilarity_vale)) {				
				termScore.add(0.0);
			} else {				
				termScore.add(cosineSimilarity_vale);
			}			
		}
        
	}
	
	private double CosineSimilarity(double[] docVector1, double[] docVector2)
	{
		double dotProduct = 0.0;
        double magnitude1 = 0.0;
        double magnitude2 = 0.0;
        double cosineSimilarity = 0.0;
        
        for (int i=0; i<docVector1.length; i++)
        {
            dotProduct += docVector1[i] * docVector2[i];
            magnitude1 += Math.pow(docVector1[i], 2);
            magnitude2 += Math.pow(docVector2[i], 2);
        }

        magnitude1 = Math.sqrt(magnitude1);
        magnitude2 = Math.sqrt(magnitude2);

        if (magnitude1 != 0.0 | magnitude2 != 0.0) {
            cosineSimilarity = dotProduct / (magnitude1 * magnitude2);
        } else {
            return 0.0;
        }
        
        return cosineSimilarity;
	}
	
	private void BubbleSort_ArrayList(ArrayList<Double> arrlist, ArrayList<String> arrlistTitle)
	{						
		double value_temp;
		String title_temp;
	
		for(int j=0; j< arrlist.size(); j++)
		{			
			for(int i=j+1; i<arrlist.size(); i++)
			{
			    if(arrlist.get(i) > arrlist.get(j)){
			    	value_temp = arrlist.get(j);
			    	arrlist.set( j, arrlist.get(i));
			    	arrlist.set( i, value_temp);
			    	
			    	title_temp = arrlistTitle.get(j);
			    	arrlistTitle.set(j, arrlistTitle.get(i));
			    	arrlistTitle.set(i, title_temp);
			   }
			}
		}
		
	}
	
	public static void main( String[] args) {
		try {
			SongSimilarity w2c = new SongSimilarity();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
