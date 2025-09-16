# The MIT License (MIT)
# Copyright © 2025 Subquery

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import copy
import os
from pathlib import Path
import torch.multiprocessing as mp
import random
import time
from fastapi.responses import StreamingResponse
from loguru import logger
import numpy as np
import bittensor as bt
import torch
import uvicorn
from multiprocessing.synchronize import Event
from common.logger import HermesLogger
from common.protocol import ChatCompletionRequest, OrganicNonStreamSynapse, OrganicStreamSynapse
import common.utils as utils
from hermes.validator.challenge_manager import ChallengeManager
from hermes.validator.api import app
from hermes.base import BaseNeuron

HermesLogger.configure_loguru(file=f"logs/hermes_validator.log")

class Validator(BaseNeuron):
    dendrite: bt.Dendrite
    hotkeys: dict[int, str]  # uid to hotkey mapping
    scores: torch.Tensor
    device: str
    _last_set_weight_time: float
    challenge_manager: ChallengeManager

    @property
    def role(self) -> str:
        return "validator"
    
    def __init__(self):
        super().__init__()
        
        self._last_set_weight_time = time.time()

        self.hotkeys = copy.deepcopy(self.settings.metagraph.hotkeys)
        self.scores = torch.zeros_like(torch.tensor(self.settings.metagraph.S), dtype=torch.float32)
        self.device = 'cpu'

        self.dendrite = bt.dendrite(wallet=self.settings.wallet)
        
        self.set_weight_interval = int(os.getenv("SET_WEIGHT_INTERVAL", 60 * 30))  # seconds
        logger.info(f"Set weight interval set to {self.set_weight_interval} seconds")

    async def start(self):
        super().start()

        self.challenge_manager = ChallengeManager(
            settings=self.settings,
            save_project_dir=Path(__file__).parent.parent / "projects" / self.role,
            uid=self.uid,
            dendrite=self.dendrite
        )

        tasks = [
            asyncio.create_task(
                self.challenge_manager.start()
            ),
            asyncio.create_task(
                self.serve_api()
            ),
            asyncio.create_task(
                self.set_weight()
            ),
        ]
        await asyncio.gather(*tasks)

    async def run_challenge(self, organic_score_queue: list, event_stop: Event):
        self.challenge_manager = ChallengeManager(
            settings=self.settings,
            save_project_dir=Path(__file__).parent.parent / "projects" / self.role,
            uid=self.uid,
            dendrite=self.dendrite,
            organic_score_queue=organic_score_queue,
            event_stop=event_stop
        )
        tasks = [
            asyncio.create_task(
                self.challenge_manager.start()
            ),
        ]
        await asyncio.gather(*tasks)

    async def run_api(self, organic_score_queue: list):
        super().start()
        self.organic_score_queue = organic_score_queue

        try:
            external_ip = utils.try_get_external_ip()
            logger.info(f"external_ip: {external_ip}")

            logger.info(f"Starting serve API on http://0.0.0.0:{self.settings.port}")
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.settings.port,
                loop="asyncio",
                reload=False,
                log_config=None,  # Disable uvicorn's default logging config
                access_log=False,  # Disable access logs to reduce noise
            )
            app.state.validator = self

            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"Failed to serve API: {e}")

    async def forward_miner(self, cid: str, body: ChatCompletionRequest):
        uids = [u for u in self.settings.miners() if u != self.uid]
        miner_uid = random.choice(uids)
        logger.info(f"[Validator] Received organic task({body.id}) cid: {cid}, body: {body}, forward to miner_uid: {miner_uid}")

        if body.stream:
            async def streamer():
                synapse = OrganicStreamSynapse(project_id=cid, completion=body)
                responses = await self.dendrite.forward(
                    axons=self.settings.metagraph.axons[miner_uid],
                    synapse=synapse,
                    deserialize=True,
                    timeout=60*3,
                    streaming=True,
                )
                async for part in responses:
                    # logger.info(f"V3 got part: {part}, type: {type(part)}")
                    yield part
            return StreamingResponse(streamer(), media_type="text/plain")

        synapse = OrganicNonStreamSynapse(project_id=cid, completion=body)
        start_time = time.perf_counter()
        response = await self.dendrite.forward(
            axons=self.settings.metagraph.axons[miner_uid],
            synapse=synapse,
            deserialize=True,
            timeout=10,
        )
        elapsed_time = time.perf_counter() - start_time
        response.elapsed_time = elapsed_time

        logger.info(f"[Validator] organic task({body.id}), miner response: {response}")

        # logger.info(f'----{response.dendrite.status_code}')
        # logger.info(f'----{response.dendrite.status_message}')
        self.organic_score_queue.append((miner_uid, response.dict()))
        return response

    async def set_weight(self):
        while True:
            await asyncio.sleep(10)
            if time.time() - self._last_set_weight_time > self.set_weight_interval:
                try:
                    scores_dict = self.challenge_manager.scorer_manager.get_last_scores()
                    uids = list(scores_dict.keys())
                    scores = list(scores_dict.values())
                    if not uids:
                        continue
                    self._set_weights(uids, scores)

                    self._last_set_weight_time = time.time()
                except Exception as e:
                    logger.error(f"Failed to set_weight: {e}")

    def _set_weights(self, uids: list[int], scores: list[float]):
        logger.info(f"set_weights for uids: {uids}, scores: {scores}")

        scattered_scores: torch.FloatTensor = self.scores.scatter(
            0, torch.tensor(uids).to(self.device), torch.tensor(scores, dtype=torch.float32).to(self.device)
        ).to(self.device)
        
        logger.info(f"scattered_scores: {scattered_scores}")

        raw_weights = torch.nn.functional.normalize(scattered_scores, p=1, dim=0)
        logger.info(f"raw_weights: {raw_weights}")

        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids = np.array(self.settings.metagraph.uids, dtype=np.int64),
                weights = raw_weights.detach().cpu().numpy().astype(np.float32),
                netuid=self.settings.netuid,
                subtensor=self.settings.subtensor,
                metagraph=self.settings.metagraph,
        )
        logger.info(f"processed_weight_uids: {processed_weight_uids}")
        logger.info(f"processed_weights: {processed_weights}")

        [suc, msg] = self.settings.subtensor.set_weights(
            wallet=self.settings.wallet,
            netuid=self.settings.netuid,
            uids=processed_weight_uids,
            weights=processed_weights,
            wait_for_finalization=False,
            version_key=10010,
        )
        logger.info(f"processed_weights: {suc, msg}")

def run_challenge(organic_score_queue: list, event_stop: Event):
    proc = mp.current_process()
    HermesLogger.configure_loguru(file=f"logs/{proc.name}.log")

    logger.info(f"run_challenge process id: {os.getpid()}")
    asyncio.run(Validator().run_challenge(organic_score_queue, event_stop))

def run_api(organic_score_queue):
    proc = mp.current_process()
    HermesLogger.configure_loguru(file=f"logs/{proc.name}.log")

    logger.info(f"run_api process id: {os.getpid()}")
    asyncio.run(Validator().run_api(organic_score_queue))

async def main():
    with mp.Manager() as manager:
        try:
            organic_score_queue = manager.list([])
            processes: list[mp.Process] = []
            event_stop = mp.Event()
        
            challenge_process = mp.Process(
                target=run_challenge,
                args=(organic_score_queue, event_stop),
                name="ChallengeProcess",
                daemon=True,
            )
            challenge_process.start()
            processes.append(challenge_process)

            api_process = mp.Process(
                target=run_api,
                args=(organic_score_queue,),
                name="APIProcess",
                daemon=True,
            )
            api_process.start()
            processes.append(api_process)

            logger.info(f"main process id: {os.getpid()}")
            while True:
                await asyncio.sleep(10)

        except KeyboardInterrupt:
            event_stop.set()
            logger.info("KeyboardInterrupt detected. Shutting down gracefully...")

        except Exception as e:
            logger.error(f"Main loop error: {e}")
            raise

        finally:
            utils.kill_process_group()

if __name__ == "__main__":
    try:
        os.setpgrp()
    except BaseException:
        logger.warning("Failed to set process group.")

    asyncio.run(main())




