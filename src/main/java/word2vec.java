
/*
 * word2vec at Song Recommendation (Similar)
 * 
 * version: November 11, 2019 04:40 PM
 * Last revision: November 12, 2019 11:27 AM
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

public class word2vec
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
	
	// Similarity
	//private ArrayList<double[]> vecValue = new ArrayList();
	//ArrayList<double[]> vecValue = new ArrayList<double[]>(); 
	private	ArrayList<java.util.Vector> vecValue = new ArrayList<java.util.Vector>();
	
	private String inputStr = "然後 呢 他們 說 你 的 心 似乎 痊癒 了 也 開始 有 個人 為 你 守護 著 我該 心安 或者 心碎 呢 然後 呢 其實 我 的 日子 也還 可以 呢 除了 回憶 肆虐 的 某些 時刻 慶幸 還有 淚水 沖淡 苦澀 而 那些 昨天 仍 繽紛 著 它們 都 有 我 細心 收藏 著 也許 你還 記得 也許 你 都 忘記 了 也 不是 那 麽 重要 了 只 期待 後來 的 你 能 快樂 那 就是 後來 的 我 最想 的 後來 的 我倆 仍 走著 只是 不再 並肩 了 朝 各自 的 人生 追尋 了 無論是 後來 故事 怎麼 麽 了 也 要 讓 後來 人生 精彩 著 後來 的 我倆 我 期待 著 淚水 中能 看見 你 真的 自由 了 親愛 的 回憶 我倆 共同 走過 的 曲折 是 那些 帶 我倆 來到 了 這 一刻 讓 珍貴 的 人生 有失 有 得 用 新 的 幸福 把 遺憾 包著 就 這 麽 朝著 未來 前進 了 有 再 多 的 不 捨 也 要 狠心 割捨 別 回頭 看 我 親愛 的 只 期待 後來 的 你 能 快樂 那 就是 後來 的 我 最想 的 後來 的 我倆 仍 走著 只是 不再 並肩 了 朝 各自 的 人生 追尋 了 無論是 後來 故事 怎麼 麽 了 也 要 讓 後來 人生 精彩 著 後來 的 我倆 我 期待 著 淚水 中能 看見 你 真的 幸福快樂 在 某處 另 一個 你 留下 了 在 那裏 另 一個 我 微笑 著 另 一個 我倆 還 深愛 著 代表 我倆 永遠 著 如果 能 這 麽 想 就夠 了 無論是 後來 故事 怎麼 麽 了 也 要 讓 後來 人生 值得 後來 的 我倆 我 期待 著 淚水 中能 看見 你 真的 自由 了";
	private double[] inputValue = new double[wordim];
	private double cosineSimilarity_vale;
	private java.util.Vector cosineSimilarity_vale_vec;
	private double[] vector_list_1;	
	private double[] vector_list_2;
	
	private java.util.Vector allValue = new java.util.Vector();
	
	public word2vec() throws Exception
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
		Read_Vec();
		Comparison(inputValue, allValue);
		
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
	
	private void Read_Vec() throws Exception 
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
	
	//private void Comparison(double[] targetVector, List tfidfDocsVector) 
	private void Comparison(double[] targetVector, java.util.Vector allValue)
	{	
		cosineSimilarity_vale_vec = new java.util.Vector();
		
		ArrayList<double[]> tfidfDocsVector = new ArrayList<double[]>();
		double[] temp;
		String[] QQ;
		for(int i=0; i<allValue.size(); i++)
		{
			temp = new double[wordim];
			//System.out.println(allValue.get(i));
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
				cosineSimilarity_vale_vec.add(0);
			} else {
				cosineSimilarity_vale_vec.add(cosineSimilarity_vale);
			}
			System.out.println(j+" "+cosineSimilarity_vale);
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
	
	public static void main( String[] args ) {
		try {
			word2vec w2c = new word2vec();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
