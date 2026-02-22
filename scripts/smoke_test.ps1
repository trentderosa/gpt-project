param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [switch]$IncludeChat
)

$ErrorActionPreference = "Stop"

function Step($msg) {
  Write-Host ""
  Write-Host "==> $msg" -ForegroundColor Cyan
}

Step "Health check"
$health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health"
$health | ConvertTo-Json -Depth 4

Step "Plans check"
$plans = Invoke-RestMethod -Method Get -Uri "$BaseUrl/plans"
$plans | ConvertTo-Json -Depth 5

$stamp = Get-Date -Format "yyyyMMddHHmmss"
$email = "smoke+$stamp@example.com"
$password = "SmokeTestPass123!"

Step "Register account"
$reg = Invoke-RestMethod -Method Post -Uri "$BaseUrl/auth/register" -ContentType "application/json" -Body (@{
  email = $email
  password = $password
} | ConvertTo-Json)
$token = $reg.token
if (-not $token) { throw "Missing auth token from register." }
Write-Host "Registered: $email"

$headers = @{ Authorization = "Bearer $token" }

Step "Auth me"
$me = Invoke-RestMethod -Method Get -Uri "$BaseUrl/auth/me" -Headers $headers
$me | ConvertTo-Json -Depth 4

Step "Create conversation"
$convo = Invoke-RestMethod -Method Post -Uri "$BaseUrl/conversations" -Headers $headers
$conversationId = $convo.conversation_id
if (-not $conversationId) { throw "Missing conversation_id." }
Write-Host "Conversation: $conversationId"

if ($IncludeChat) {
  Step "Chat request"
  $chat = Invoke-RestMethod -Method Post -Uri "$BaseUrl/chat" -Headers (@{
    Authorization = "Bearer $token"
    "Content-Type" = "application/json"
  }) -Body (@{
    message = "Say hello in one sentence."
    conversation_id = $conversationId
    use_web_search = $false
  } | ConvertTo-Json)
  $chat | ConvertTo-Json -Depth 4
} else {
  Write-Host ""
  Write-Host "Skipping /chat (use -IncludeChat to test model path)." -ForegroundColor Yellow
}

Step "Smoke test completed"
Write-Host "All core endpoints responded successfully." -ForegroundColor Green
