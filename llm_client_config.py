from pydantic import Field, BaseModel, ConfigDict


class LlmClientConfig(BaseModel):
    """General LLM client configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: str = Field(
        description="API key for the LLM service.",
    )

    model: str = "openai/gpt-oss-120b"

    politeness_delay_sec: int = Field(
        default=15,
        gt=0,
        lt=60,
        description="Delay between LLM requests to avoid rate limiting.",
    )
    
    max_output_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum number of output tokens to generate.",
    )
    