import { config as dotenvConfig } from "dotenv";
import fs from "fs/promises";
import puppeteer from "puppeteer";
import TurndownService from "turndown";
import { MarkdownTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { PineconeStore } from "langchain/vectorstores";
import { PineconeClient } from "@pinecone-database/pinecone";

dotenvConfig();

const client = new PineconeClient();
let pineconeIndex;
const turndownService = new TurndownService();

if (!process.env.PINECONE_API_KEY) {
  throw new Error("PINECONE_API_KEY is not set");
}
if (!process.env.PINECONE_ENVIRONMENT) {
  throw new Error("PINECONE_ENVIRONMENT is not set");
}

export const main = async (index) => {
  pineconeIndex = index;

  const markdowns = await scrape_all_pages();
  await generate_embeddings(markdowns.join("\n\n"));
};

async function get_all_pages_from_cache() {
  return await fs.readFile("generated/all.txt", "utf8");
}

async function scrape_all_pages() {
  const { urls } = JSON.parse(await fs.readFile("scripts/pages.json", "utf8"));
  console.log(`Got ${urls.length} urls ready to scrape`);

  const browser = await puppeteer.launch();
  const markdowns = [];
  for (const url of urls) {
    try {
      const markdown = await scrape_page(url, browser);
      markdowns.push(markdown);
    } catch (e) {
      console.log(`Error scraping ${url}`);
    }
  }
  await browser.close();

  console.log(`Got ${markdowns.length} markdowns ready to save`);

  try {
    await fs.readdir("./generated");
  } catch (e) {
    await fs.mkdir("./generated");
  }
  await fs.writeFile("./generated/all.txt", markdowns.join("\n\n"));
  console.log(`Saved all markdowns to ./generated/all.txt`);

  return markdowns;
}

async function generate_embeddings(markdowns) {
  const textSplitter = new MarkdownTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 20,
  });
  const embeddings = await textSplitter.splitText(markdowns);
  console.log(`Got ${embeddings.length} embeddings ready to save`);

  const embeddingModel = new OpenAIEmbeddings({ maxConcurrency: 5 });

  await PineconeStore.fromTexts(embeddings, [], embeddingModel, {
    pineconeIndex,
  });
  console.log(`Saved embeddings to pinecone index ${pineconeIndex}`);
}

async function scrape_page(url, browser) {
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle2' });
  
    const pageContent = await page.evaluate(() => {
      return document.documentElement.innerText;
    });
  
    return pageContent;
  }
  

async function scrape_researchr_page(url, browser) {
  const page = await browser.newPage();
  await page.setJavaScriptEnabled(false);
  await page.goto(url);

  const element = await page.waitForSelector("#content > div.row > div", {
    timeout: 100,
  });

  if (!element) {
    throw new Error("Could not find element");
  }

  await element.evaluate((element) => {
    const elements = element.querySelectorAll(
      "*:not(p, h1, h2, h3, h4, h5, h6, li, blockquote, pre, code, table, dl, div)"
    );
    for (let i = 0; i < elements.length; i++) {
      elements[i].parentNode?.removeChild(elements[i]);
    }
  });

  const html_of_element = await element.evaluate(
    (element) => element.innerHTML
  );
  return turndownService.turndown(html_of_element);
}
