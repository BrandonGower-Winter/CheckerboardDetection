import ECAgent.Core as core
from ECAgent.Environments import PositionComponent, SpaceWorld


class DataComponent(core.Component):
    SELF_FULFILLING = 0
    SELF_DEFEATING = 1
    SELF_REINFORCING = 2
    SELF_ADVERSARIAL = 3

    def __init__(self, agent, model: core.Model, label, weight: float = 1.0, sigma: float = 1,
                 weight_delta: float = 0.01, move_delta: float = 0.01,
                 x_drift: float = 0.0, y_drift: float = 0.0,
                 feedback_type: list = None, decay_feedback: bool = False):
        super().__init__(agent, model)

        self.label = label
        self.sigma = sigma
        self.init_weight = weight
        self.weight = weight

        self.move_delta = move_delta
        self.weight_delta = weight_delta

        self.x_drift = x_drift
        self.y_drift = y_drift

        if feedback_type is None:
            self.feedback_type = [0, 0, 0, 0]
        else:
            assert len(feedback_type) == 4  # TODO Proper Error Message
            self.feedback_type = feedback_type

        self.decay_feedback = decay_feedback

    def __str__(self):
        return f"Pos: {self.agent[PositionComponent].x}, Label: {self.label}, Weight: {self.weight}, " \
               f"Sigma: {self.sigma}, wDelta: {self.weight_delta}"


class DataAgent(core.Agent):
    def __init__(self, id: str, model: core.Model, label, x: float = 0.0, y: float = 0.0,
                 weight: float = 1.0, sigma: float = 1.0, weight_delta: float = 0.01,
                 move_delta: float = 0.01, x_drift: float = 0.0, y_drift: float = 0.0,
                 feedback_type: list = None, decay_feedback: bool = False):
        super().__init__(id, model)
        self.add_component(PositionComponent(self, model, x, y))
        self.add_component(DataComponent(self, model, label, weight, sigma,
                                         weight_delta, move_delta, x_drift, y_drift, feedback_type,
                                         decay_feedback))


class PerformativeSystem(core.System):
    def __init__(self, id: str, model: core.Model, predictor,
                 priority: int = 0, frequency: int = 1, samples: int = 1):
        super().__init__(id, model, priority=priority, frequency=frequency)

        self._samples = samples
        self._stream = []
        self._predictor = predictor

    def execute(self):

        # Get All Agents
        choices = self.model.environment.get_agents()
        weights = [a[DataComponent].weight for a in choices]
        t = self.model.systems.timestep
        # For samples
        for _ in range(self._samples):
            # Generate Data Point

            if sum(weights) < 0.0000001:  # If weights are near 0.0, randomly select instance
                centroid = self.model.random.choices(choices)[0]
            else:
                centroid = self.model.random.choices(choices, weights=weights)[0]

            dc = centroid[DataComponent]
            pc = centroid[PositionComponent]
            data_point = (self.model.random.gauss(pc.x, dc.sigma), self.model.random.gauss(pc.y, dc.sigma))

            #print(data_point[0])
            y_hat = self._predictor(data_point, dc.label, t)
            self._stream.append((data_point, dc.label, y_hat, t))

            # Move Model
            if y_hat == dc.label:  # TODO Is this a realistic assumption
                dist = ((data_point[0] - pc.x) ** 2 + (data_point[1] - pc.y) ** 2) ** 0.5
                x_move = (data_point[0] - pc.x) / dist * dc.move_delta
                y_move = (data_point[1] - pc.y) / dist * dc.move_delta

                if dc.feedback_type[DataComponent.SELF_FULFILLING] == 1:
                    dc.weight += dc.weight_delta

                if dc.feedback_type[DataComponent.SELF_DEFEATING] == 1:
                    dc.weight -= dc.weight_delta
                    dc.weight = max(0.0, dc.weight)

                if dc.feedback_type[DataComponent.SELF_REINFORCING] == 1:
                    pc.x += x_move
                    pc.y += y_move

                if dc.feedback_type[DataComponent.SELF_ADVERSARIAL] == 1:
                    pc.x -= x_move
                    pc.y -= y_move
            elif dc.decay_feedback:
                if dc.feedback_type[DataComponent.SELF_DEFEATING] == 1:
                    dc.weight += dc.weight_delta

                if dc.feedback_type[DataComponent.SELF_FULFILLING] == 1:
                    dc.weight -= dc.weight_delta
                    dc.weight = max(dc.init_weight, dc.weight)
                # TODO Think about how decay affects other feedback types

    def get_stream(self):
        return iter(self._stream)

    def clear_stream(self):
        self._stream.clear()

    def change_predictor(self, predictor):
        self._predictor = predictor


class IntrinsicDriftSystem(core.System):

    DRIFT_GRADUAL = 0
    DRIFT_SUDDEN = 1

    DRIFT_COLLECTIVE = 0
    DRIFT_INDIVIDUAL = 1

    def __init__(self, id: str, model: core.Model, drift_reset: int = -1, drift_magnitude: float = 1.0,
                 priority: int = 0, frequency: int = 1, drift_type: int = DRIFT_GRADUAL,
                 drift_mode: int = DRIFT_COLLECTIVE):
        super().__init__(id, model, priority=priority, frequency=frequency)
        self.drift_reset = drift_reset
        self._drift_counter = 0
        self.drift_magnitude = drift_magnitude

        self._drift_type = drift_type
        self._drift_mode = drift_mode
        self._set_all_random_drifts()

    def _apply_sudden_drift(self):
        for agent in self.model.environment.get_agents():
            pc = agent[PositionComponent]
            pc.x = self.model.random.uniform(-1.0, 1.0)

    def _set_all_random_drifts(self):

        if self._drift_mode == IntrinsicDriftSystem.DRIFT_COLLECTIVE:
            x_drift = self.model.random.uniform(-1.0, 1.0)
            y_drift = self.model.random.uniform(-1.0, 1.0)

        for agent in self.model.environment.get_agents():
            dc = agent[DataComponent]

            if self._drift_mode == IntrinsicDriftSystem.DRIFT_INDIVIDUAL:
                x_drift = self.model.random.uniform(-1.0, 1.0)
                y_drift = self.model.random.uniform(-1.0, 1.0)

            dc.x_drift = x_drift
            dc.y_drift = y_drift

    def execute(self):

        if self._drift_type == IntrinsicDriftSystem.DRIFT_SUDDEN:
            self._apply_sudden_drift()
            return

        self._drift_counter += 1

        # Reset the drift
        if self.drift_reset > 0 and self._drift_counter % self.drift_reset == 0:
            self._drift_counter = 0
            self._set_all_random_drifts()

        for agent in self.model.environment.get_agents():
            dc = agent[DataComponent]
            pc = agent[PositionComponent]
            pc.x += dc.x_drift * self.drift_magnitude
            pc.y += dc.y_drift * self.drift_magnitude


class AgentStream(core.Model):

    def __init__(self, predictor, samples: int = 1, drift_reset: int = -1,
                 drift_mode: int = IntrinsicDriftSystem.DRIFT_COLLECTIVE,
                 drift_magnitude: float = 0.0, seed: int = None):
        super().__init__(seed=seed)

        # Add Systems
        self.systems.add_system(PerformativeSystem("PERF", self, predictor, samples=samples))
        self.systems.add_system(IntrinsicDriftSystem("INT", self, drift_reset=drift_reset,
                                                     drift_magnitude=drift_magnitude, drift_mode=drift_mode))

    def get_stream(self):
        return self.systems['PERF'].get_stream()

    def clear_stream(self):
        self.systems['PERF'].clear_stream()

    def change_predictor(self, predictor):
        self.systems['PERF'].change_predictor(predictor)

    def __str__(self):
        res = "Agent Stream:"

        for agent in self.environment:
            res += f"\n{agent.id}: {agent[DataComponent]}"

        return res
