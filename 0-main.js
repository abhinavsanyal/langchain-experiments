
import { PineconeClient } from "@pinecone-database/pinecone";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import * as dotenv from "dotenv";
import { createPineconeIndex } from "./1-createPineconeIndex.js";
import { updatePinecone , updatePineconeWithDataFromWebPages} from "./2-updatePinecone.js";
import { queryPineconeVectorStoreAndQueryLLM, queryPineconeVectorStoreAndQueryLLMForWebSites } from "./3-queryPineconeAndQueryGPT.js";

dotenv.config();

const loader = new DirectoryLoader("./documents", {
    ".txt": (path) => new TextLoader(path),
    ".pdf": (path) => new PDFLoader(path),
});
const docs = await loader.load();
// 8. Set up variables for the filename, question, and index settings
const question = "what is the testimonial of Praveen dayal?";
const indexName = "sensai";
const vectorDimension = 1536;
// 9. Initialize Pinecone client with API key and environment
const client = new PineconeClient();
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
// 10. Run the main async function
(async () => {
// 11. Check if Pinecone index exists and create if necessary
  await createPineconeIndex(client, indexName, vectorDimension);
// 12. Update Pinecone vector store with document embeddings
  // await updatePinecone(client, indexName, docs);
  await updatePineconeWithDataFromWebPages(client, indexName);
// 13. Query Pinecone vector store and GPT model for an answer
  // await queryPineconeVectorStoreAndQueryLLM(client, indexName, question);
  await queryPineconeVectorStoreAndQueryLLMForWebSites(client, indexName, question);
})();