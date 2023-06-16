import { OpenAIChat } from "langchain/llms/openai";
import { loadSummarizationChain } from "langchain/chains";
import { Document } from "langchain/document";
import { transcriptionSegments } from "../data.js";


export const run = async () => {
  const model = new OpenAIChat({ temperature: 0, azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY });
  const chain = loadSummarizationChain(model, { type: "refine", verbose: true });
  const docs: Document[] = [];
  let batch: string[] = [];
  for (const segment of transcriptionSegments) {
    batch.push(segment.text);
    if (batch.length === 10) {
      docs.push(new Document({ pageContent: batch.join('\n\n') }) );
      batch = [];
    }
  }

  const res = await chain.call({
    input_documents: docs,
  });
  console.log(res);
};
