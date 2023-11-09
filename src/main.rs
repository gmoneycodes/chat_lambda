use lambda_http::{lambda, IntoResponse, Request, RequestExt, Response};
use aws_sdk_dynamodb::{Client as DynamoClient, Error as DynamoError, model::AttributeValue};
use aws_config::meta::region::RegionProviderChain;
use serde_json::json;
use serde::{Deserialize, Serialize};
use std::env;
use hyper::{Client as HyperClient, Request as HyperRequest, Body as HyperBody, header};
use hyper_tls::HttpsConnector;
use std::sync::Arc;
use tokio;

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

// Helper function to read preambles from dynamodb
async fn fetch_preamble_from_dynamodb(dynamo_client: &DynamoClient, character_id: &str) -> Result<String, DynamoError> {
    let get_item_output = dynamo_client.get_item()
        .table_name("Characters")
        .key("CharacterID", AttributeValue::S(character_id.to_owned()))
        .send()
        .await?;

    if let Some(item) = get_item_output.item {
        if let Some(preamble) = item.get("Prompt").and_then(AttributeValue::as_s) {
            return Ok(preamble.to_owned());
        }
    }

    Err(DynamoError::Unhandled("No preamble found for the given CharacterID".into()))
}
async fn function_handler(event: Request) -> Result<impl IntoResponse, Error> {
    // Load environment variables
    dotenv::dotenv().ok();

    let oai_token = env::var("OAI_TOKEN").expect("Expected an OAI_TOKEN in the environment");
    let auth_header_val = format!("Bearer {}", oai_token);

    let https = HttpsConnector::new();
    let client = Client::builder().build::<_, HyperBody>(https);

    // This will be deprecated April 2024
    let uri = "https://api.openai.com/v1/engines/text-davinci-003/completions";

    // Extract the user's text from the query string
    let user_text = event
        .query_string_parameters()
        .first("text")
        .unwrap_or_default();

    // Extract the character ID from the query string
    let character_id = event
        .query_string_parameters()
        .first("character_id")
        .unwrap_or_default();

    // Fetch preamble from DynamoDB
    let selected_preamble = match fetch_preamble_from_dynamodb(dynamo_client, character_id).await {
        Ok(preamble) => preamble,
        Err(e) => {
            eprintln!("DynamoDB error: {:?}", e);
            return Err(Error::from("Failed to fetch preamble from DynamoDB"));
        },
    };

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
async fn main() -> Result<(), lambda_http::Error> {
    // Set up the region provider chain to default to 'us-west-2' if no other region is found.
    let region_provider = RegionProviderChain::default_provider().or_else("us-west-2");

    // Load the AWS configuration using the region provider chain.
    let shared_config = aws_config::from_env().region(region_provider).load().await;

    // Create a DynamoDB client with the loaded configuration.
    let dynamo_client = DynamoClient::new(&shared_config);

    // Wrap the DynamoDB client in an Arc for sharing across invocations.
    let dynamo_client = Arc::new(dynamo_client);

    // Use the lambda! macro to run your function handler
    lambda!(|event, _context| {
        let client = Arc::clone(&dynamo_client);
        async move {
            function_handler(event, client).await
        }
    });

    Ok(())
}
}