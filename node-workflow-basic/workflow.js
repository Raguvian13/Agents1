/**
 * Basic LangChain.js Workflow: Email Composer Pipeline
 *
 * This script demonstrates a sequential 3-step workflow:
 * Key points + tone → [Draft Email] → [Check for Issues] → [Final Version]
 *
 * Key concepts:
 * - Prompt templates with variable substitution
 * - Sequential chain execution (pipe operator)
 * - Error handling with retry logic
 * - Passing multiple variables down a composed chain
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda } from "@langchain/core/runnables";

// --- Configuration ---
// Uses Groq API (free, ultra-fast inference) — get a key at https://console.groq.com

if (!process.env.GROQ_API_KEY) {
  console.error("ERROR: GROQ_API_KEY is not set.");
  console.error("Get a free key at https://console.groq.com");
  console.error('Set it with: export GROQ_API_KEY="your-key-here"');
  console.error("Or create a .env file with: GROQ_API_KEY=your-key-here");
  process.exit(1);
}

const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  apiKey: process.env.GROQ_API_KEY,
});
const outputParser = new StringOutputParser();

// --- Step 1: Draft Email ---

const draftEmailPrompt = ChatPromptTemplate.fromTemplate(
  `Draft an email based on the following key points:
{keyPoints}

Ensure the tone of the email is: {tone}.`
);

const draftEmailChain = draftEmailPrompt.pipe(llm).pipe(outputParser);

// --- Step 2: Check for Issues ---

const checkIssuesPrompt = ChatPromptTemplate.fromTemplate(
  `Review the following email draft. Identify any issues related to clarity, grammar, tone inconsistency, or missing information.
List the issues concisely. If there are no major issues, reply with "No major issues found."

Email Draft:
{draft}`
);

const checkIssuesChain = checkIssuesPrompt.pipe(llm).pipe(outputParser);

// --- Step 3: Final Version ---

const finalVersionPrompt = ChatPromptTemplate.fromTemplate(
  `You are an expert editor. Here is an original email draft and a list of issues found during review.
Rewrite the email to fix all the issues and improve its overall flow. Output ONLY the final polished email.

Original Draft:
{draft}

Issues to fix:
{issues}`
);

const finalVersionChain = finalVersionPrompt.pipe(llm).pipe(outputParser);

// --- Error Handling: Retry with Exponential Backoff ---

async function callWithRetry(chain, inputs, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await chain.invoke(inputs);
    } catch (error) {
      const message = error.message || String(error);
      const isRetryable =
        message.includes("429") ||
        message.includes("500") ||
        message.toLowerCase().includes("timeout");

      if (isRetryable && attempt < maxRetries) {
        const delay = Math.pow(2, attempt) * 1000;
        console.log(
          `  Attempt ${attempt} failed: ${message.slice(0, 60)}...`
        );
        console.log(`  Retrying in ${delay / 1000}s...`);
        await new Promise((r) => setTimeout(r, delay));
      } else {
        throw error;
      }
    }
  }
}

// --- Run the Workflow ---

async function runWorkflow(keyPoints, tone) {
  console.log("=".repeat(60));
  console.log(`Email Composer Workflow`);
  console.log(`Key Points: "${keyPoints}"\nTone: "${tone}"`);
  console.log("=".repeat(60));

  // Step 1
  console.log("\nSTEP 1: Drafting initial email...");
  let draft;
  try {
    draft = await callWithRetry(draftEmailChain, { keyPoints, tone });
    console.log(draft);
  } catch (error) {
    console.error(`FATAL: Initial draft generation failed: ${error.message}`);
    console.error("Cannot continue without a draft.");
    return;
  }

  // Step 2
  console.log("\n" + "-".repeat(60));
  console.log("STEP 2: Checking for issues...");
  let issues;
  try {
    // Validate intermediate output before passing to next step
    if (draft.trim().length < 20) {
      throw new Error(
        "Draft is too short — LLM may have returned incomplete output."
      );
    }
    issues = await callWithRetry(checkIssuesChain, { draft });
    console.log(issues);
  } catch (error) {
    console.error(`FATAL: Issue checking failed: ${error.message}`);
    return;
  }

  // Step 3
  console.log("\n" + "-".repeat(60));
  console.log("STEP 3: Creating final version...");
  try {
    const finalEmail = await callWithRetry(finalVersionChain, { draft, issues });
    console.log(finalEmail);
  } catch (error) {
    // Final version is critical here, unlike the summary in the previous script
    console.error(`FATAL: Final version generation failed: ${error.message}`);
  }

  console.log("\n" + "=".repeat(60));
  console.log("Workflow complete!");
}

// --- Bonus: Composed Single Chain ---

async function runComposedWorkflow(keyPoints, tone) {
  console.log("\n" + "=".repeat(60));
  console.log("Running as a single composed chain...");
  console.log("=".repeat(60));

  // Compose all steps into one chain. 
  // We use RunnableLambda to pass the 'draft' forward alongside 'issues' so Step 3 has both.
  const fullWorkflow = draftEmailChain
    .pipe(new RunnableLambda({ func: (draft) => ({ draft }) }))
    .pipe(new RunnableLambda({ 
        func: async ({ draft }) => {
            const issues = await checkIssuesChain.invoke({ draft });
            return { draft, issues };
        }
    }))
    .pipe(finalVersionChain);

  try {
    const finalEmail = await fullWorkflow.invoke({ keyPoints, tone });
    console.log("\nFinal Email:");
    console.log(finalEmail);
  } catch (error) {
    console.error(`Workflow failed: ${error.message}`);
  }
}

// --- Main ---

const keyPoints = process.argv[2] || "Project ISAAC is delayed by 2 weeks due to supply chain issues. Ask for a brief meeting next Tuesday.";
const tone = process.argv[3] || "Professional, apologetic, and solution-oriented";

await runWorkflow(keyPoints, tone);
await runComposedWorkflow(keyPoints, tone);