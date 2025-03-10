if __name__ == "__main__":
    pipeline = NLPPipeline(nlp_config)

    # Add observers for monitoring
    pipeline.attach(LoggingObserver())
    pipeline.attach(MetricsObserver())

    # Execute with retry logic
    retry(pipeline.run, retries=3, delay=10)