import net.dv8tion.jda.api.JDA;
import net.dv8tion.jda.api.JDABuilder;
import net.dv8tion.jda.api.OnlineStatus;
import net.dv8tion.jda.api.entities.Activity;
import net.dv8tion.jda.api.entities.Message;
import net.dv8tion.jda.api.entities.MessageChannel;
import net.dv8tion.jda.api.events.message.MessageReceivedEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;
import opennlp.tools.doccat.*;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.*;
import opennlp.tools.util.model.ModelUtil;

import javax.security.auth.login.LoginException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
// NLP is natural language processing.

public class OpenNLPChatBot extends ListenerAdapter {

	private static Map<String, String> questionAnswer = new HashMap<>();

	private static SentenceDetectorME sentenceCategories;
	private static TokenizerME sentenceTokens;
	private static POSTaggerME myCategorizer2;
	private static LemmatizerME myCategorizer3;
//read all the bin files to create models
	static {
		try {
			InputStream english_sentence_input = new FileInputStream("en-sent.bin");
			sentenceCategories = new SentenceDetectorME(new SentenceModel(english_sentence_input));

			InputStream english_token_input = new FileInputStream("en-token.bin");
			sentenceTokens = new TokenizerME(new TokenizerModel(english_token_input));

			InputStream part_of_speech_input = new FileInputStream("en-pos-maxent.bin");
			myCategorizer2 = new POSTaggerME(new POSModel(part_of_speech_input));

			InputStream lemma_token_input = new FileInputStream("en-lemmatizer.bin");
			myCategorizer3 = new LemmatizerME(new LemmatizerModel(lemma_token_input));


		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	//Define answers for each given category.
	static {
		questionAnswer.put("greeting", "Hello, how can I help you?");
		questionAnswer.put("product-inquiry",
				"Product is a Porsche 911 GT2 RS.");
		questionAnswer.put("price-inquiry", "Price is $300,000");
		questionAnswer.put("conversation-continue", "What else can I help you with?");
		questionAnswer.put("nice-ending", "Goodbye!");
		questionAnswer.put("cfa", "Chik-Fil-a");
	}

    private OpenNLPChatBot() throws IOException {
    }


    public static void main(String[] args) throws IOException, LoginException {

		// Create the bot with your own bot token from discord developer website
		JDA jda = new JDABuilder("BOT_TOKEN").addEventListeners(new OpenNLPChatBot())
				.setActivity(Activity.watching("The Gate"))
				.setStatus(OnlineStatus.IDLE)
				.build();
	}

    private DoccatModel model = trainCategorizeModel();

	@Override
	public void onMessageReceived(MessageReceivedEvent event)
	{
		//This makes the bot not respond to itself
		if(event.getAuthor().isBot()) {
			return;
		}

		//Gets the user inout as a string
		Message msg = event.getMessage();
		String msgString = msg.getContentRaw();

		// Break users chat input into sentences using sentence detection.
		String[] sentences = new String[0];
		try {
			sentences = breakSentences(msgString);
		} catch (IOException e) {
			e.printStackTrace();
		}

		StringBuilder answer = new StringBuilder();
		boolean conversationComplete = false;

		// Loop through sentences.
		for (String sentence : sentences) {

			// Separate words from each sentence using tokenizer.
			String[] tokens = new String[0];
			try {
				tokens = tokenizeSentence(sentence);
			} catch (IOException e) {
				e.printStackTrace();
			}

			// Tag separated words with POS tags to understand their grammatical structure.
			String[] posTags = new String[0];
			try {
				posTags = detectPOSTags(tokens);
			} catch (IOException e) {
				e.printStackTrace();
			}

			// Lemma each word so that its easy to categorize.
			String[] lemmas = new String[0];
			try {
				lemmas = lemmatizeTokens(tokens, posTags);
			} catch (IOException e) {
				e.printStackTrace();
			}

			// Determine BEST category using lemma tokens
			String category = null;
			try {
				category = detectCategory(model, lemmas);
			} catch (IOException e) {
				e.printStackTrace();
			}

			// Get predefined answer from given category & add to answer.
			answer.append(" ").append(questionAnswer.get(category));

			// If category conversation-complete, we will end chat conversation.
			if ("conversation-complete".equals(category)) {
				conversationComplete = true;

			}
		}

		//Put the bot's answer into the discord chat
		MessageChannel channel = event.getChannel();
		channel.sendMessage(answer).queue();

		if (conversationComplete) {
			//We can tell the bot to do something when the conversation is over
		}

		//Ping the bot
		if (msg.getContentRaw().equals("!ping"))
		{
			long time = System.currentTimeMillis();
			channel.sendMessage("Pong!")
					.queue(response -> {
						response.editMessageFormat("Pong: %d ms", System.currentTimeMillis() - time).queue();
					});
		}
	}


	//Train model as per the category in the sample file
	private static DoccatModel trainCategorizeModel() throws IOException {
		InputStreamFactory inputStreamFactory = new MarkableFileInputStreamFactory(new File("response-categories.txt"));
		ObjectStream<String> lineStream = new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
		ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

		DoccatFactory factory = new DoccatFactory(new FeatureGenerator[] { new BagOfWordsFeatureGenerator() });

		TrainingParameters params = ModelUtil.createDefaultTrainingParameters();
		params.put(TrainingParameters.CUTOFF_PARAM, 0);
		params.put(TrainingParameters.ITERATIONS_PARAM, 500);

		// Train a model with classifications from above file.
		return DocumentCategorizerME.train("en", sampleStream, params, factory);
	}


	//Detect category using given token.
	private static String detectCategory(DoccatModel model, String[] finalTokens) throws IOException {

		// Initialize document category tool
		DocumentCategorizerME categoryTool = new DocumentCategorizerME(model);

		// Get best possible category.
		double[] probabilitiesOfOutcomes = categoryTool.categorize(finalTokens);
		String category = categoryTool.getBestCategory(probabilitiesOfOutcomes);
		System.out.println("Category: " + category);

		return category;
	}


	//Break data into sentences using sentence detection feature of Apache OpenNLP.
	private static String[] breakSentences(String data) throws IOException {
			String[] sentences = sentenceCategories.sentDetect(data);
			System.out.println("Sentence Detection: " + String.join(" | ", sentences));

			return sentences;
	}


	 //Break sentence into words & punctuation marks using tokenizer feature of Apache OpenNLP.
	private static String[] tokenizeSentence(String sentence) throws IOException {
			String[] tokens = sentenceTokens.tokenize(sentence);
			System.out.println("Tokenizer : " + String.join(" | ", tokens));

			return tokens;
	}


	//Find part-of-speech or POS tags of all tokens using POS tagger feature of Apache OpenNLP.
	private static String[] detectPOSTags(String[] tokens) throws IOException {
			// Tag sentence.
			String[] posTokens = myCategorizer2.tag(tokens);
			System.out.println("POS Tags : " + String.join(" | ", posTokens));

			return posTokens;
	}

	 // Find lemma of tokens using lemmatizer feature of Apache OpenNLP.
	private static String[] lemmatizeTokens(String[] tokens, String[] posTags)
			throws IOException {

			String[] lemmaTokens = myCategorizer3.lemmatize(tokens, posTags);
			System.out.println("Lemmatizer : " + String.join(" | ", lemmaTokens));

			return lemmaTokens;
	}

}