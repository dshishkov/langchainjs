import { OpenAIChat } from "langchain/llms/openai";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { transcriptionSegments } from "../data.js";

export const run = async () => {
  // In this example, we use a `MapReduceDocumentsChain` specifically prompted to summarize a set of documents.
  const text = transcriptionSegments.map((s) => s.text).join("\n");
  const model = new OpenAIChat({ temperature: 0, azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY });
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200, separators: ["\n"] });
  const docs = await textSplitter.createDocuments([text]);

  // This convenience function creates a document chain prompted to summarize a set of documents.
  const chain = loadSummarizationChain(model, { type: "refine", verbose: true });
  const res = await chain.call({
    input_documents: docs,
  });
  console.log({ res });
};
