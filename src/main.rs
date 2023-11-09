use lambda_http::{run, service_fn, Body, Error, IntoResponse, Request, Response};
use serde_json::{json, Value};
use serde_derive::{Deserialize, Serialize};
use std::env;
use hyper::{Client, Request as HyperRequest, Body as HyperBody, header};
use hyper_tls::HttpsConnector;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

// Struct to fetch OpenAI API Response
#[derive(Deserialize, Debug)]
struct OAIResponse {
    id: Option<String>,
    object: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    choices: Vec<OAIChoices>,
}

// Struct to capture options/choices
#[derive(Deserialize, Debug)]
struct OAIChoices {
    text: String,
    index: u8,
    logprobs: Option<u8>,
    finish_reason: String,
}

// Request Struct
#[derive(Serialize, Debug)]
struct OAIRequest {
    prompt: String,
    max_tokens: u16,
}

// Helper function to read preambles from file
fn read_preambles_from_file() -> io::Result<Vec<String>> {
    let file = File::open("preambles.txt")?;
    let reader = BufReader::new(file);
    reader.lines().collect()
}

async fn function_handler(event: Request) -> Result<impl IntoResponse, Error> {
    // Load environment variables
    dotenv::dotenv().ok();

    let oai_token = env::var("OAI_TOKEN").expect("Expected an OAI_TOKEN in the environment");
    let auth_header_val = format!("Bearer {}", oai_token);

    let https = HttpsConnector::new();
    let client = Client::builder().build::<_, HyperBody>(https);

    let uri = "https://api.openai.com/v1/engines/text-davinci-003/completions";

    // Extract the user's text from the query string
    let user_text = event
        .query_string_parameters()
        .first("text")
        .unwrap_or_default();

    // Read preambles from file
    let preambles = read_preambles_from_file().unwrap_or_else(|_| vec![]);
    // Select a random preamble for simplicity, you can implement your own logic
    let selected_preamble = preambles.get(0).cloned().unwrap_or_default();

    let oai_request = OAIRequest {
        prompt: format!("{} {}", selected_preamble, user_text),
        max_tokens: 1000,
    };

    let request_body = serde_json::to_vec(&oai_request)?;
    let req = HyperRequest::post(uri)
        .header(header::AUTHORIZATION, &auth_header_val)
        .header(header::CONTENT_TYPE, "application/json")
        .body(HyperBody::from(request_body))
        .expect("Failed to build request");

    // Send the request to the OpenAI API
    let response = client.request(req).await?;
    let body = hyper::body::aggregate(response).await?;
    let response_bytes = body.to_bytes().await.map_err(|e| {
        Error::from(e)
    })?;

    let json: OAIResponse = serde_json::from_slice(&response_bytes)?;

    // Construct the response
    let response_text = json.choices.get(0)
        .map(|choice| choice.text.trim())
        .unwrap_or_default()
        .to_string();

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "text/plain")
        .body(response_text.into())
        .expect("Failed to render response"))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(function_handler)).await?;
    Ok(())
}